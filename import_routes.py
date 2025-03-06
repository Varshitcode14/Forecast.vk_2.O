import os
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import pytz
import re
from models import db, Customer, Vendor, Product, Sale, SaleItem, Purchase, PurchaseItem
import time
import logging
from functools import wraps
from sqlalchemy import create_engine, or_
from sqlalchemy.pool import QueuePool
import gc  # Garbage collection
import json
import threading
from flask import jsonify, Response, stream_with_context
from flask_login import current_user
import uuid  # For generating truly unique IDs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a Blueprint for import routes
import_bp = Blueprint('import_bp', __name__)

# Define IST timezone
IST = pytz.timezone('Asia/Kolkata')

# Simple in-memory status tracking
import_status = {}

# Database connection retry decorator with exponential backoff
def db_retry(max_retries=5, initial_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    
            logger.error(f"All database retry attempts failed: {str(last_exception)}")
            raise last_exception
            
        return wrapper
    return decorator

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['xlsx', 'xls', 'csv']

# Helper function to create upload directory if it doesn't exist
def ensure_upload_dir():
    upload_dir = os.path.join(current_app.root_path, 'uploads')
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    return upload_dir

# Helper function to parse date based on format with better error handling
def parse_date(date_str, date_format):
    try:
        if isinstance(date_str, (datetime, pd.Timestamp)):
            return date_str
            
        if pd.isna(date_str):
            raise ValueError("Date string is empty or NaN")
            
        date_str = str(date_str).strip()
        
        if date_format == 'DD/MM/YYYY':
            return datetime.strptime(date_str, '%d/%m/%Y')
        elif date_format == 'MM/DD/YYYY':
            return datetime.strptime(date_str, '%m/%d/%Y')
        elif date_format == 'YYYY-MM-DD':
            return datetime.strptime(date_str, '%Y-%m-%d')
        elif date_format == 'DD-MM-YYYY':
            return datetime.strptime(date_str, '%d-%m-%Y')
        elif date_format == 'MM-DD-YYYY':
            return datetime.strptime(date_str, '%m-%d-%Y')
        else:
            # Try to infer the format
            for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y']:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Could not parse date: {date_str}")
    except Exception as e:
        logger.error(f"Error parsing date '{date_str}': {str(e)}")
        raise ValueError(f"Error parsing date '{date_str}': {str(e)}")

# Helper function to generate a unique product ID from product name
def generate_product_id(product_name):
    # Remove special characters and spaces, convert to uppercase
    base_id = re.sub(r'[^a-zA-Z0-9]', '', product_name).upper()
    
    # Take first 4 characters (or less if name is shorter)
    prefix = base_id[:min(4, len(base_id))]
    
    # Add a random 4-digit number
    import random
    suffix = str(random.randint(1000, 9999))
    
    return f"P{prefix}{suffix}"

# Completely rewritten function to generate truly unique GST IDs
def generate_unique_gst_id(entity_type, user_id):
    """
    Generate a globally unique GST ID that doesn't conflict with any existing IDs
    
    Args:
        entity_type: String, either 'CUST' for customer or 'VEND' for vendor
        user_id: The user ID to associate with this entity
        
    Returns:
        A unique GST ID string
    """
    try:
        # Create a base prefix that includes user ID to avoid conflicts between users
        base_prefix = f"{entity_type}{user_id}"
        
        # Generate a short random suffix (8 chars from uuid4)
        random_suffix = str(uuid.uuid4()).replace('-', '')[:8].upper()
        
        # Combine to create a unique ID
        unique_id = f"{base_prefix}_{random_suffix}"
        
        # Double-check it doesn't exist in the database
        if entity_type == 'CUST':
            exists = Customer.query.filter_by(gst_id=unique_id).first() is not None
        else:  # VEND
            exists = Vendor.query.filter_by(gst_id=unique_id).first() is not None
            
        # In the extremely unlikely case of a collision, try again
        if exists:
            return generate_unique_gst_id(entity_type, user_id)
            
        logger.info(f"Generated unique GST ID: {unique_id}")
        return unique_id
        
    except Exception as e:
        logger.error(f"Error generating unique GST ID: {str(e)}")
        # Fallback to timestamp-based ID to ensure uniqueness
        timestamp = int(time.time())
        return f"{entity_type}{user_id}_{timestamp}"

# Add this function at the top of your file, after the imports
def is_production():
    """Check if we're running in production environment"""
    return os.environ.get('RENDER', False)

# Modify the process_dataframe_in_chunks function
def process_dataframe_in_chunks(df, chunk_size=None):
    """Process dataframe in chunks to avoid memory issues"""
    # Use smaller chunks in production
    if chunk_size is None:
        chunk_size = 10 if is_production() else 25
    
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx].copy()  # Use copy to avoid memory leaks
        yield chunk
        # Force garbage collection after each chunk
        gc.collect()

# Add these functions for database optimization

def optimize_db_for_import():
    """Optimize database for bulk imports"""
    try:
        # Only run PostgreSQL specific optimizations in production
        if is_production():
            # Create a new connection to run these commands
            engine = db.engine
            connection = engine.raw_connection()
            cursor = connection.cursor()
            
            # Disable triggers temporarily
            cursor.execute("SET session_replication_role = 'replica';")
            
            # Increase work memory for sorting
            cursor.execute("SET work_mem = '64MB';")
            
            # Disable synchronous commit temporarily
            cursor.execute("SET synchronous_commit = OFF;")
            
            connection.commit()
            cursor.close()
            connection.close()
            
            logger.info("Database optimized for import")
        else:
            # SQLite optimizations
            engine = db.engine
            connection = engine.raw_connection()
            cursor = connection.cursor()
            
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL;")
            
            # Disable synchronous writes temporarily for speed
            cursor.execute("PRAGMA synchronous=OFF;")
            
            # Increase cache size
            cursor.execute("PRAGMA cache_size=10000;")
            
            connection.commit()
            cursor.close()
            connection.close()
            
            logger.info("SQLite database optimized for import")
        
        return True
    except Exception as e:
        logger.error(f"Error optimizing database: {str(e)}")
        return False

def restore_db_after_import():
    """Restore database settings after import"""
    try:
        if is_production():
            # PostgreSQL restore
            engine = db.engine
            connection = engine.raw_connection()
            cursor = connection.cursor()
            
            # Re-enable triggers
            cursor.execute("SET session_replication_role = 'origin';")
            
            # Reset work memory
            cursor.execute("RESET work_mem;")
            
            # Re-enable synchronous commit
            cursor.execute("RESET synchronous_commit;")
            
            connection.commit()
            cursor.close()
            connection.close()
            
            logger.info("PostgreSQL database settings restored after import")
        else:
            # SQLite restore
            engine = db.engine
            connection = engine.raw_connection()
            cursor = connection.cursor()
            
            # Re-enable synchronous writes
            cursor.execute("PRAGMA synchronous=NORMAL;")
            
            # Reset cache size
            cursor.execute("PRAGMA cache_size=-2000;")
            
            connection.commit()
            cursor.close()
            connection.close()
            
            logger.info("SQLite database settings restored after import")
        
        return True
    except Exception as e:
        logger.error(f"Error restoring database settings: {str(e)}")
        return False

# Function to save import status
def save_import_status(import_id, status_data):
    import_status[import_id] = status_data
    logger.info(f"Status updated for {import_id}: {status_data['message']}")

# Function to get import status
def get_import_status(import_id):
    return import_status.get(import_id, {
        'status': 'error',
        'message': 'Import not found'
    })

# Main import page route
@import_bp.route('/import', methods=['GET'])
def import_data():
    return render_template('import_data.html')

# Sales import routes
@import_bp.route('/import/sales', methods=['GET'])
def import_sales():
    return render_template('import_sales.html', step=1)

@import_bp.route('/import/sales/validate', methods=['POST'])
def validate_sales_file():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('import_bp.import_sales'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('import_bp.import_sales'))
    
    if file and allowed_file(file.filename):
        try:
            # Save the file
            upload_dir = ensure_upload_dir()
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            saved_filename = f"sales_{timestamp}_{filename}"
            file_path = os.path.join(upload_dir, saved_filename)
            file.save(file_path)
            
            # Validate file structure
            try:
                # Read the file
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                # Check if the file has the required columns
                required_columns = ['Invoice Number', 'Date', 'Customer Name', 'Product Name', 'Quantity', 'Total Amount']
                
                # Check if all required columns are present
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    flash(f'Missing required columns: {", ".join(missing_columns)}', 'error')
                    os.remove(file_path)  # Remove the invalid file
                    return redirect(url_for('import_bp.import_sales'))
                
                # Check if file is not empty
                if len(df) == 0:
                    flash('File is empty', 'error')
                    os.remove(file_path)
                    return redirect(url_for('import_bp.import_sales'))
                
                # File is valid, proceed to date format selection
                return render_template('import_sales.html', step=2, file_path=file_path)
                
            except Exception as e:
                logger.error(f"Error validating file: {str(e)}")
                flash(f'Error validating file: {str(e)}', 'error')
                if os.path.exists(file_path):
                    os.remove(file_path)
                return redirect(url_for('import_bp.import_sales'))
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            flash('Error saving file. Please try again.', 'error')
            return redirect(url_for('import_bp.import_sales'))
    else:
        flash('Invalid file type. Please upload an Excel (.xlsx, .xls) or CSV file.', 'error')
        return redirect(url_for('import_bp.import_sales'))

@import_bp.route('/import/sales/preview', methods=['POST'])
@db_retry(max_retries=3, initial_delay=2)
def preview_sales_data():
    file_path = request.form.get('file_path')
    date_format = request.form.get('date_format')
    
    if not file_path or not os.path.exists(file_path):
        flash('File not found. Please upload again.', 'error')
        return redirect(url_for('import_bp.import_sales'))
    
    if not date_format:
        flash('Please select a date format', 'error')
        return render_template('import_sales.html', step=2, file_path=file_path)
    
    try:
        # Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Replace the date parsing loop with:
        df['parsed_date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        date_warnings = df['parsed_date'].isna().any()
        
        # Prepare preview data
        preview_data = []
        
        for _, row in df.head(10).iterrows():
            try:
                if pd.notna(row['parsed_date']):
                    date_display = row['parsed_date'].strftime('%Y-%m-%d')
                else:
                    date_display = str(row['Date'])
                
                # Calculate unit price and tax
                quantity = float(row['Quantity'])
                total_amount = float(row['Total Amount'])
                gst_percentage = 18.0  # Default GST
                
                # Calculate pre-tax amount: total_amount / (1 + gst_percentage/100)
                pre_tax_amount = total_amount / (1 + gst_percentage/100)
                tax_amount = total_amount - pre_tax_amount
                unit_price = pre_tax_amount / quantity
                
                preview_data.append({
                    'invoice_number': row['Invoice Number'],
                    'date': date_display,
                    'customer_name': row['Customer Name'],
                    'product_name': row['Product Name'],
                    'quantity': int(quantity),
                    'unit_price': round(unit_price, 2),
                    'gst_amount': round(tax_amount, 2),
                    'total_amount': round(total_amount, 2)
                })
            except Exception as e:
                logger.error(f"Error processing row: {str(e)}")
                continue
        
        # Check if products exist
        product_names = df['Product Name'].unique()
        existing_products = Product.query.filter(Product.name.in_(product_names)).filter_by(user_id=current_user.id).all()
        existing_product_names = [p.name for p in existing_products]
        missing_products = [name for name in product_names if name not in existing_product_names]
        
        # Check if customers exist
        customer_names = df['Customer Name'].unique()
        existing_customers = Customer.query.filter(Customer.name.in_(customer_names)).filter_by(user_id=current_user.id).all()
        existing_customer_names = [c.name for c in existing_customers]
        missing_customers = [name for name in customer_names if name not in existing_customer_names]
        
        return render_template('import_sales.html', 
                             step=3, 
                             file_path=file_path, 
                             date_format=date_format,
                             preview_data=preview_data,
                             total_rows=len(df),
                             date_warnings=date_warnings,
                             missing_products=missing_products,
                             missing_customers=missing_customers)
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('import_bp.import_sales'))

# Add this debugging function to help diagnose import issues
def debug_dataframe(df, import_id):
    """Log dataframe information for debugging"""
    try:
        logger.info(f"Import {import_id} - DataFrame info:")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"First 5 rows: {df.head(5).to_dict('records')}")
        
        # Check for NaN values
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            logger.info(f"NaN counts: {nan_counts.to_dict()}")
            
        return True
    except Exception as e:
        logger.error(f"Error debugging dataframe: {str(e)}")
        return False

# Helper function to find or create a vendor with proper error handling
def find_or_create_vendor(vendor_name, user_id, import_id):
    """Find an existing vendor or create a new one with proper error handling"""
    try:
        # IMPORTANT: Only search within the current user's vendors
        # First, try to find by exact name match for this user
        vendor = Vendor.query.filter_by(name=vendor_name, user_id=user_id).first()
        
        if vendor:
            logger.info(f"Import {import_id} - Found existing vendor: {vendor_name} (ID: {vendor.id}) for user {user_id}")
            return vendor, False  # Found existing vendor
        
        # Try case-insensitive search as fallback (still within user's data)
        vendor = Vendor.query.filter(
            Vendor.name.ilike(f"%{vendor_name}%"), 
            Vendor.user_id == user_id
        ).first()
        
        if vendor:
            logger.info(f"Import {import_id} - Found similar vendor: {vendor.name} for {vendor_name} (ID: {vendor.id}) for user {user_id}")
            return vendor, False  # Found similar vendor
        
        # Create a new vendor with a guaranteed unique GST ID
        gst_id = generate_unique_gst_id("VEND", user_id)
        
        vendor = Vendor(
            user_id=user_id,
            name=vendor_name,
            gst_id=gst_id,
            location="Unknown",  # Placeholder location
            contact_person="",
            phone="",
            about=f"Imported from Excel on {datetime.now().strftime('%Y-%m-%d')}"
        )
        
        db.session.add(vendor)
        db.session.flush()  # Get ID without committing
        
        logger.info(f"Import {import_id} - Created new vendor: {vendor_name} with GST ID: {gst_id} (ID: {vendor.id}) for user {user_id}")
        return vendor, True  # Created new vendor
        
    except Exception as e:
        logger.error(f"Error finding/creating vendor {vendor_name} for user {user_id}: {str(e)}")
        raise  # Re-raise to be handled by the caller

# Helper function to find or create a customer with proper error handling
def find_or_create_customer(customer_name, user_id, import_id):
    """Find an existing customer or create a new one with proper error handling"""
    try:
        # IMPORTANT: Only search within the current user's customers
        # First, try to find by exact name match for this user
        customer = Customer.query.filter_by(name=customer_name, user_id=user_id).first()
        
        if customer:
            logger.info(f"Import {import_id} - Found existing customer: {customer_name} (ID: {customer.id}) for user {user_id}")
            return customer, False  # Found existing customer
        
        # Try case-insensitive search as fallback (still within user's data)
        customer = Customer.query.filter(
            Customer.name.ilike(f"%{customer_name}%"), 
            Customer.user_id == user_id
        ).first()
        
        if customer:
            logger.info(f"Import {import_id} - Found similar customer: {customer.name} for {customer_name} (ID: {customer.id}) for user {user_id}")
            return customer, False  # Found similar customer
        
        # Create a new customer with a guaranteed unique GST ID
        gst_id = generate_unique_gst_id("CUST", user_id)
        
        customer = Customer(
            user_id=user_id,
            name=customer_name,
            gst_id=gst_id,
            location="Unknown",  # Placeholder location
            contact_person="",
            phone="",
            about=f"Imported from Excel on {datetime.now().strftime('%Y-%m-%d')}"
        )
        
        db.session.add(customer)
        db.session.flush()  # Get ID without committing
        
        logger.info(f"Import {import_id} - Created new customer: {customer_name} with GST ID: {gst_id} (ID: {customer.id}) for user {user_id}")
        return customer, True  # Created new customer
        
    except Exception as e:
        logger.error(f"Error finding/creating customer {customer_name} for user {user_id}: {str(e)}")
        raise  # Re-raise to be handled by the caller

# Modify the process_sales_import_task function to fix small file handling
def process_sales_import_task(import_id, file_path, date_format, create_missing):
    try:
        # Get the current user ID from the import_id (format: "sales_TIMESTAMP_USER_ID")
        user_id = import_id.split('_')[-1]
        
        # Optimize database for import
        optimize_db_for_import()
        
        # Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Debug the dataframe
        debug_dataframe(df, import_id)
        
        # Parse dates first
        df['parsed_date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        
        total_rows = len(df)
        logger.info(f"Processing {total_rows} rows for sales import {import_id}")
        
        # Initialize results
        import_results = {
            'status': 'processing',
            'progress': 10,
            'message': f"Processing {total_rows} sales records...",
            'success': 0,
            'warnings': 0,
            'errors': 0,
            'new_customers': [],
            'new_products': [],
            'errors_list': []
        }
        save_import_status(import_id, import_results)
        
        # Get existing products for this user
        existing_products = {p.name: p for p in Product.query.filter_by(user_id=user_id).all()}
        
        # Process data in smaller chunks
        chunk_size = min(10, max(1, total_rows // 10))  # Adjust chunk size based on total rows
        chunks = list(process_dataframe_in_chunks(df, chunk_size))
        total_chunks = len(chunks)
        
        logger.info(f"Import {import_id} - Processing {total_chunks} chunks with size {chunk_size}")
        
        for chunk_idx, chunk in enumerate(chunks):
            # Add this at the beginning of each chunk processing loop in both functions
            db_session_valid = True
            try:
                # Update progress
                progress = 10 + int(80 * chunk_idx / total_chunks)
                import_results['progress'] = progress
                import_results['message'] = f"Processing chunk {chunk_idx + 1} of {total_chunks}..."
                save_import_status(import_id, import_results)
                
                # Log chunk info
                logger.info(f"Import {import_id} - Processing chunk {chunk_idx + 1}/{total_chunks} with {len(chunk)} rows")
                
                # Group by invoice number and customer name
                grouped = chunk.groupby(['Invoice Number', 'Customer Name'])
                
                for (invoice_number, customer_name), group_data in grouped:
                    try:
                        logger.info(f"Import {import_id} - Processing invoice {invoice_number} for customer {customer_name}")
                        
                        # Get or create customer
                        if create_missing:
                            customer, is_new = find_or_create_customer(customer_name, user_id, import_id)
                            if is_new:
                                import_results['new_customers'].append(customer_name)
                        else:
                            # Only look for existing customer
                            customer = Customer.query.filter_by(name=customer_name, user_id=user_id).first()
                            if not customer:
                                import_results['errors'] += 1
                                import_results['errors_list'].append(f"Customer not found: {customer_name}")
                                logger.warning(f"Import {import_id} - Customer not found and not creating: {customer_name}")
                                continue
                        
                        # Get the first valid date from the group
                        valid_dates = group_data['parsed_date'].dropna().tolist()
                        if valid_dates:
                            sale_date = valid_dates[0]
                        else:
                            sale_date = datetime.now()
                            logger.warning(f"Import {import_id} - Using current date for invoice {invoice_number} due to missing date")
                        
                        # Calculate total amount for the sale
                        total_amount = 0
                        delivery_charges = 0  # Default to 0, can be updated if in the data
                        
                        # Create sale
                        sale = Sale(
                            user_id=user_id,  # Associate with the current user
                            customer_id=customer.id,
                            sale_date=sale_date,
                            delivery_charges=delivery_charges,
                            total_amount=0  # Will update after calculating items
                        )
                        db.session.add(sale)
                        db.session.flush()  # Get ID without committing
                        logger.info(f"Import {import_id} - Created sale record for invoice {invoice_number}")
                        
                        # Process each item in the sale
                        for _, row in group_data.iterrows():
                            product_name = row['Product Name']
                            
                            # Get or create product
                            if product_name in existing_products:
                                product = existing_products[product_name]
                                logger.info(f"Import {import_id} - Using existing product: {product_name}")
                            elif create_missing:
                                # Generate a unique product ID
                                product_id = generate_product_id(product_name)
                                
                                # Create a new product with placeholder data
                                product = Product(
                                    user_id=user_id,  # Associate with the current user
                                    product_id=product_id,
                                    name=product_name,
                                    quantity=0,  # Initial quantity
                                    cost_per_unit=0,  # Will be updated based on sales/purchases
                                    specifications=""
                                )
                                db.session.add(product)
                                db.session.flush()  # Get ID without committing
                                existing_products[product_name] = product
                                import_results['new_products'].append(product_name)
                                logger.info(f"Import {import_id} - Created new product: {product_name}")
                            else:
                                import_results['errors'] += 1
                                import_results['errors_list'].append(f"Product not found: {product_name}")
                                logger.warning(f"Import {import_id} - Product not found and not creating: {product_name}")
                                continue
                            
                            # Calculate item price
                            try:
                                quantity = int(float(row['Quantity']))
                                total_amount_item = float(row['Total Amount'])
                                gst_percentage = 18.0  # Default GST
                                discount_percentage = 0.0  # Default discount
                                
                                # Calculate pre-tax amount
                                pre_tax_amount = total_amount_item / (1 + gst_percentage/100)
                                unit_price = pre_tax_amount / quantity if quantity > 0 else 0
                                
                                # Check if enough stock
                                if product.quantity < quantity:
                                    import_results['warnings'] += 1
                                    import_results['errors_list'].append(f"Warning: Not enough stock for product {product_name}. Current: {product.quantity}, Required: {quantity}")
                                    logger.warning(f"Import {import_id} - Not enough stock for product {product_name}")
                                
                                # Update product quantity
                                product.quantity -= quantity
                                
                                # Create sale item
                                sale_item = SaleItem(
                                    sale_id=sale.id,
                                    product_id=product.id,
                                    quantity=quantity,
                                    gst_percentage=gst_percentage,
                                    discount_percentage=discount_percentage,
                                    unit_price=unit_price,
                                    total_price=total_amount_item
                                )
                                db.session.add(sale_item)
                                logger.info(f"Import {import_id} - Added sale item: {product_name}, qty: {quantity}")
                                
                                # Add to total amount
                                total_amount += total_amount_item
                            except Exception as e:
                                import_results['errors'] += 1
                                import_results['errors_list'].append(f"Error processing item {product_name}: {str(e)}")
                                logger.error(f"Import {import_id} - Error processing item {product_name}: {str(e)}")
                                continue
                        
                        # Update sale total amount
                        sale.total_amount = total_amount
                        
                        import_results['success'] += 1
                        logger.info(f"Import {import_id} - Successfully processed invoice {invoice_number}, total: {total_amount}")
                        
                    except Exception as e:
                        logger.error(f"Error processing sale: {str(e)}")
                        import_results['errors'] += 1
                        import_results['errors_list'].append(f"Error importing sale: {str(e)}")
                        # Check if this is a transaction error that requires rollback
                        if "transaction has been rolled back" in str(e) or "IntegrityError" in str(e):
                            db.session.rollback()
                            db_session_valid = False
                            logger.warning("Transaction rolled back, will skip remaining items in this chunk")
                            break  # Exit the grouped data loop
                        continue
                
                # After the grouped data loop in both functions
                if not db_session_valid:
                    logger.warning(f"Import {import_id} - Skipping commit for chunk {chunk_idx + 1} due to transaction errors")
                    continue  # Skip to the next chunk
                
                # Commit after each chunk - update this in both functions
                if db_session_valid:
                    try:
                        db.session.commit()
                        logger.info(f"Import {import_id} - Committed chunk {chunk_idx + 1}")
                    except Exception as e:
                        db.session.rollback()
                        logger.error(f"Error committing chunk {chunk_idx + 1}: {str(e)}")
                        import_results['errors'] += 1
                        import_results['errors_list'].append(f"Error committing chunk: {str(e)}")
                else:
                    # Reset the session for the next chunk
                    db.session.rollback()
                
                # Force garbage collection after each chunk
                gc.collect()
                
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error processing chunk: {str(e)}")
                import_results['errors'] += 1
                import_results['errors_list'].append(f"Error processing chunk: {str(e)}")
        
        # Update final status
        import_results['status'] = 'completed'
        import_results['progress'] = 100
        import_results['message'] = f"Import completed successfully. Imported {import_results['success']} sales."
        save_import_status(import_id, import_results)
        logger.info(f"Import {import_id} - Completed with {import_results['success']} successes, {import_results['warnings']} warnings, {import_results['errors']} errors")
        
        # Clean up the file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Restore database settings
        restore_db_after_import()
            
    except Exception as e:
        logger.error(f"Error in sales import: {str(e)}")
        import_results = {
            'status': 'error',
            'message': f"Error: {str(e)}",
            'errors_list': [str(e)]
        }
        save_import_status(import_id, import_results)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Make sure to restore database settings even on error
        restore_db_after_import()

# Modify the process_sales_import_background function to properly use app context
def process_sales_import_background(app, import_id, file_path, date_format, create_missing):
    """Process sales import in a background thread with proper app context"""
    with app.app_context():
        try:
            process_sales_import_task(import_id, file_path, date_format, create_missing)
        except Exception as e:
            logger.error(f"Background task error: {str(e)}")
            import_results = {
                'status': 'error',
                'message': f"Error: {str(e)}",
                'errors_list': [str(e)]
            }
            save_import_status(import_id, import_results)

# Update the process_sales_import route to use the background function
@import_bp.route('/import/sales/process', methods=['POST'])
def process_sales_import():
    file_path = request.form.get('file_path')
    date_format = request.form.get('date_format')
    create_missing = request.form.get('create_missing') == 'yes'
    
    if not file_path or not os.path.exists(file_path):
        flash('File not found. Please upload again.', 'error')
        return redirect(url_for('import_bp.import_sales'))
    
    try:
        # Generate a unique import ID that includes the user ID
        import_id = f"sales_{datetime.now().strftime('%Y%m%d%H%M%S')}_{current_user.id}"
        
        # Initialize status
        save_import_status(import_id, {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting import...'
        })
        
        # Get the current app
        app = current_app._get_current_object()
        
        # Start the import process in a background thread with app context
        thread = threading.Thread(
            target=process_sales_import_background,
            args=(app, import_id, file_path, date_format, create_missing)
        )
        thread.daemon = True
        thread.start()
        
        # Return the status page immediately
        return render_template('import_status.html', import_id=import_id, import_type='sales')
        
    except Exception as e:
        logger.error(f"Error starting import: {str(e)}")
        flash(f'Error starting import: {str(e)}', 'error')
        return redirect(url_for('import_bp.import_sales'))

# Purchases import routes
@import_bp.route('/import/purchases', methods=['GET'])
def import_purchases():
    return render_template('import_purchases.html', step=1)

@import_bp.route('/import/purchases/validate', methods=['POST'])
def validate_purchases_file():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('import_bp.import_purchases'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('import_bp.import_purchases'))
    
    if file and allowed_file(file.filename):
        # Save the file
        upload_dir = ensure_upload_dir()
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        saved_filename = f"purchases_{timestamp}_{filename}"
        file_path = os.path.join(upload_dir, saved_filename)
        file.save(file_path)
        
        # Validate file structure
        try:
            # Read the file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Check if the file has the required columns
            required_columns = ['Order ID', 'Date', 'Vendor Name', 'Product Name', 'Quantity', 'Total Amount']
            
            # Check if all required columns are present
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                flash(f'Missing required columns: {", ".join(missing_columns)}', 'error')
                os.remove(file_path)  # Remove the invalid file
                return redirect(url_for('import_bp.import_purchases'))
            
            # Check if file is not empty
            if len(df) == 0:
                flash('File is empty', 'error')
                os.remove(file_path)
                return redirect(url_for('import_bp.import_purchases'))
            
            # File is valid, proceed to date format selection
            return render_template('import_purchases.html', step=2, file_path=file_path)
            
        except Exception as e:
            flash(f'Error validating file: {str(e)}', 'error')
            if os.path.exists(file_path):
                os.remove(file_path)
            return redirect(url_for('import_bp.import_purchases'))
    else:
        flash('Invalid file type. Please upload an Excel (.xlsx, .xls) or CSV file.', 'error')
        return redirect(url_for('import_bp.import_purchases'))

@import_bp.route('/import/purchases/preview', methods=['POST'])
@db_retry(max_retries=3, initial_delay=2)
def preview_purchases_data():
    file_path = request.form.get('file_path')
    date_format = request.form.get('date_format')
    
    if not file_path or not os.path.exists(file_path):
        flash('File not found. Please upload again.', 'error')
        return redirect(url_for('import_bp.import_purchases'))
    
    if not date_format:
        flash('Please select a date format', 'error')
        return render_template('import_purchases.html', step=2, file_path=file_path)
    
    try:
        # Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Check for date parsing issues
        date_warnings = False
        parsed_dates = []
        
        for date_str in df['Date']:
            try:
                parsed_date = parse_date(str(date_str), date_format)
                parsed_dates.append(parsed_date)
            except ValueError as e:
                logger.warning(f"Date parsing warning: {str(e)}")
                date_warnings = True
                parsed_dates.append(None)
        
        df['parsed_date'] = parsed_dates
        
        # Prepare preview data
        preview_data = []
        
        for _, row in df.head(10).iterrows():
            try:
                if pd.notna(row['parsed_date']):
                    date_display = row['parsed_date'].strftime('%Y-%m-%d')
                else:
                    date_display = str(row['Date'])
                
                # Calculate unit price and tax
                quantity = float(row['Quantity'])
                total_amount = float(row['Total Amount'])
                gst_percentage = 18.0  # Default GST
                
                # Calculate pre-tax amount: total_amount / (1 + gst_percentage/100)
                pre_tax_amount = total_amount / (1 + gst_percentage/100)
                tax_amount = total_amount - pre_tax_amount
                unit_price = pre_tax_amount / quantity
                
                preview_data.append({
                    'order_id': row['Order ID'],
                    'date': date_display,
                    'vendor_name': row['Vendor Name'],
                    'product_name': row['Product Name'],
                    'quantity': int(quantity),
                    'unit_price': round(unit_price, 2),
                    'gst_amount': round(tax_amount, 2),
                    'total_amount': round(total_amount, 2),
                    'status': 'Delivered'  # Default status
                })
            except Exception as e:
                logger.error(f"Error processing row: {str(e)}")
                continue
        
        # Check if products exist
        product_names = df['Product Name'].unique()
        existing_products = Product.query.filter(Product.name.in_(product_names)).filter_by(user_id=current_user.id).all()
        existing_product_names = [p.name for p in existing_products]
        missing_products = [name for name in product_names if name not in existing_product_names]
        
        # In the preview_purchases_data function, replace the vendor check with:
        # Check if vendors exist
        vendor_names = df['Vendor Name'].unique()
        existing_vendors = Vendor.query.filter_by(user_id=current_user.id).all()
        existing_vendor_names = [v.name for v in existing_vendors]
        missing_vendors = [name for name in vendor_names if name not in existing_vendor_names]
        
        return render_template('import_purchases.html', 
                              step=3, 
                              file_path=file_path, 
                              date_format=date_format,
                              preview_data=preview_data,
                              total_rows=len(df),
                              date_warnings=date_warnings,
                              missing_products=missing_products,
                              missing_vendors=missing_vendors)
    
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('import_bp.import_purchases'))

# Process purchases import
def process_purchases_import_task(import_id, file_path, date_format, create_missing):
    try:
        # Get the current user ID from the import_id (format: "purchases_TIMESTAMP_USER_ID")
        user_id = import_id.split('_')[-1]
        
        # Optimize database for import
        optimize_db_for_import()
        
        # Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Debug the dataframe
        debug_dataframe(df, import_id)
        
        # Parse dates first
        df['parsed_date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        
        total_rows = len(df)
        logger.info(f"Processing {total_rows} rows for purchases import {import_id}")
        
        # Initialize results
        import_results = {
            'status': 'processing',
            'progress': 10,
            'message': f"Processing {total_rows} purchase records...",
            'success': 0,
            'warnings': 0,
            'errors': 0,
            'new_vendors': [],
            'new_products': [],
            'errors_list': []
        }
        save_import_status(import_id, import_results)
        
        # Get existing products for this user
        existing_products = {p.name: p for p in Product.query.filter_by(user_id=user_id).all()}
        
        # Process data in smaller chunks
        chunk_size = min(10, max(1, total_rows // 10))  # Adjust chunk size based on total rows
        chunks = list(process_dataframe_in_chunks(df, chunk_size))
        total_chunks = len(chunks)
        
        logger.info(f"Import {import_id} - Processing {total_chunks} chunks with size {chunk_size}")
        
        for chunk_idx, chunk in enumerate(chunks):
            # Add this at the beginning of each chunk processing loop in both functions
            db_session_valid = True
            try:
                # Update progress
                progress = 10 + int(80 * chunk_idx / total_chunks)
                import_results['progress'] = progress
                import_results['message'] = f"Processing chunk {chunk_idx + 1} of {total_chunks}..."
                save_import_status(import_id, import_results)
                
                # Log chunk info
                logger.info(f"Import {import_id} - Processing chunk {chunk_idx + 1}/{total_chunks} with {len(chunk)} rows")
                
                # Group by order ID and vendor name
                grouped = chunk.groupby(['Order ID', 'Vendor Name'])
                
                for (order_id, vendor_name), group_data in grouped:
                    try:
                        logger.info(f"Import {import_id} - Processing order {order_id} from vendor {vendor_name}")
                        
                        # Get or create vendor
                        if create_missing:
                            try:
                                logger.info(f"Import {import_id} - Attempting to find or create vendor: {vendor_name}")
                                vendor, is_new = find_or_create_vendor(vendor_name, user_id, import_id)
                                if is_new:
                                    import_results['new_vendors'].append(vendor_name)
                                    logger.info(f"Import {import_id} - Successfully created new vendor: {vendor_name}")
                            except Exception as e:
                                logger.error(f"Import {import_id} - Error creating vendor {vendor_name}: {str(e)}")
                                import_results['errors'] += 1
                                import_results['errors_list'].append(f"Error creating vendor {vendor_name}: {str(e)}")
                                continue
                        else:
                            # Only look for existing vendor
                            vendor = Vendor.query.filter_by(name=vendor_name, user_id=user_id).first()
                            if not vendor:
                                import_results['errors'] += 1
                                import_results['errors_list'].append(f"Vendor not found: {vendor_name}")
                                logger.warning(f"Import {import_id} - Vendor not found and not creating: {vendor_name}")
                                continue
                        
                        # Get the first valid date from the group
                        valid_dates = group_data['parsed_date'].dropna().tolist()
                        if valid_dates:
                            purchase_date = valid_dates[0]
                        else:
                            purchase_date = datetime.now()
                            logger.warning(f"Import {import_id} - Using current date for order {order_id} due to missing date")
                        
                        # Calculate total amount for the purchase
                        total_amount = 0
                        
                        # Create purchase
                        purchase = Purchase(
                            user_id=user_id,  # Associate with the current user
                            vendor_id=vendor.id,
                            purchase_date=purchase_date,
                            order_id=order_id,
                            status="Delivered",  # Default status
                            total_amount=0  # Will update after calculating items
                        )
                        db.session.add(purchase)
                        db.session.flush()  # Get ID without committing
                        logger.info(f"Import {import_id} - Created purchase record for order {order_id}")
                        
                        # Process each item in the purchase
                        for _, row in group_data.iterrows():
                            product_name = row['Product Name']
                            
                            # Get or create product
                            if product_name in existing_products:
                                product = existing_products[product_name]
                                logger.info(f"Import {import_id} - Using existing product: {product_name}")
                            elif create_missing:
                                # Generate a unique product ID
                                product_id = generate_product_id(product_name)
                                
                                # Create a new product with placeholder data
                                product = Product(
                                    user_id=user_id,  # Associate with the current user
                                    product_id=product_id,
                                    name=product_name,
                                    quantity=0,  # Initial quantity
                                    cost_per_unit=0,  # Will be updated based on purchases
                                    specifications=""
                                )
                                db.session.add(product)
                                db.session.flush()  # Get ID without committing
                                existing_products[product_name] = product
                                import_results['new_products'].append(product_name)
                                logger.info(f"Import {import_id} - Created new product: {product_name}")
                            else:
                                import_results['errors'] += 1
                                import_results['errors_list'].append(f"Product not found: {product_name}")
                                logger.warning(f"Import {import_id} - Product not found and not creating: {product_name}")
                                continue
                            
                            # Calculate item price
                            try:
                                quantity = int(float(row['Quantity']))
                                total_amount_item = float(row['Total Amount'])
                                gst_percentage = 18.0  # Default GST
                                
                                # Calculate pre-tax amount
                                pre_tax_amount = total_amount_item / (1 + gst_percentage/100)
                                unit_price = pre_tax_amount / quantity if quantity > 0 else 0
                                
                                # Update product quantity and cost_per_unit
                                product.quantity += quantity
                                
                                # Update cost_per_unit as weighted average
                                if product.cost_per_unit > 0:
                                    old_total_value = product.cost_per_unit * (product.quantity - quantity)
                                    new_value = unit_price * quantity
                                    product.cost_per_unit = (old_total_value + new_value) / product.quantity
                                else:
                                    product.cost_per_unit = unit_price
                                
                                # Create purchase item
                                purchase_item = PurchaseItem(
                                    purchase_id=purchase.id,
                                    product_id=product.id,
                                    quantity=quantity,
                                    gst_percentage=gst_percentage,
                                    unit_price=unit_price,
                                    total_price=total_amount_item
                                )
                                db.session.add(purchase_item)
                                logger.info(f"Import {import_id} - Added purchase item: {product_name}, qty: {quantity}")
                                
                                # Add to total amount
                                total_amount += total_amount_item
                            except Exception as e:
                                import_results['errors'] += 1
                                import_results['errors_list'].append(f"Error processing item {product_name}: {str(e)}")
                                logger.error(f"Import {import_id} - Error processing item {product_name}: {str(e)}")
                                continue
                        
                        # Update purchase total amount
                        purchase.total_amount = total_amount
                        
                        import_results['success'] += 1
                        logger.info(f"Import {import_id} - Successfully processed order {order_id}, total: {total_amount}")
                        
                    except Exception as e:
                        logger.error(f"Error processing purchase: {str(e)}")
                        import_results['errors'] += 1
                        import_results['errors_list'].append(f"Error importing purchase: {str(e)}")
                        # Check if this is a transaction error that requires rollback
                        if "transaction has been rolled back" in str(e) or "IntegrityError" in str(e):
                            db.session.rollback()
                            db_session_valid = False
                            logger.warning("Transaction rolled back, will skip remaining items in this chunk")
                            break  # Exit the grouped data loop
                        continue
                
                # After the grouped data loop in both functions
                if not db_session_valid:
                    logger.warning(f"Import {import_id} - Skipping commit for chunk {chunk_idx + 1} due to transaction errors")
                    continue  # Skip to the next chunk
                
                # Commit after each chunk - update this in both functions
                if db_session_valid:
                    try:
                        db.session.commit()
                        logger.info(f"Import {import_id} - Committed chunk {chunk_idx + 1}")
                    except Exception as e:
                        db.session.rollback()
                        logger.error(f"Error committing chunk {chunk_idx + 1}: {str(e)}")
                        import_results['errors'] += 1
                        import_results['errors_list'].append(f"Error committing chunk: {str(e)}")
                else:
                    # Reset the session for the next chunk
                    db.session.rollback()
                
                # Force garbage collection after each chunk
                gc.collect()
                
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error processing chunk: {str(e)}")
                import_results['errors'] += 1
                import_results['errors_list'].append(f"Error processing chunk: {str(e)}")
        
        # Update final status
        import_results['status'] = 'completed'
        import_results['progress'] = 100
        import_results['message'] = f"Import completed successfully. Imported {import_results['success']} purchases."
        save_import_status(import_id, import_results)
        logger.info(f"Import {import_id} - Completed with {import_results['success']} successes, {import_results['warnings']} warnings, {import_results['errors']} errors")
        
        # Clean up the file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Restore database settings
        restore_db_after_import()
            
    except Exception as e:
        logger.error(f"Error in purchases import: {str(e)}")
        import_results = {
            'status': 'error',
            'message': f"Error: {str(e)}",
            'errors_list': [str(e)]
        }
        save_import_status(import_id, import_results)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Make sure to restore database settings even on error
        restore_db_after_import()

# Modify the process_purchases_import_background function similarly
def process_purchases_import_background(app, import_id, file_path, date_format, create_missing):
    """Process purchases import in a background thread with proper app context"""
    with app.app_context():
        try:
            process_purchases_import_task(import_id, file_path, date_format, create_missing)
        except Exception as e:
            logger.error(f"Background task error: {str(e)}")
            import_results = {
                'status': 'error',
                'message': f"Error: {str(e)}",
                'errors_list': [str(e)]
            }
            save_import_status(import_id, import_results)

# Update the process_purchases_import route similarly
@import_bp.route('/import/purchases/process', methods=['POST'])
def process_purchases_import():
    file_path = request.form.get('file_path')
    date_format = request.form.get('date_format')
    create_missing = request.form.get('create_missing') == 'yes'
    
    logger.info(f"Processing purchases import with create_missing={create_missing} for user_id={current_user.id}")
    
    if not file_path or not os.path.exists(file_path):
        flash('File not found. Please upload again.', 'error')
        return redirect(url_for('import_bp.import_purchases'))
    
    try:
        # Generate a unique import ID that includes the user ID
        import_id = f"purchases_{datetime.now().strftime('%Y%m%d%H%M%S')}_{current_user.id}"
        
        # Initialize status
        save_import_status(import_id, {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting import...'
        })
        
        # Get the current app
        app = current_app._get_current_object()
        
        # Start the import process in a background thread with app context
        thread = threading.Thread(
            target=process_purchases_import_background,
            args=(app, import_id, file_path, date_format, create_missing)
        )
        thread.daemon = True
        thread.start()
        
        # Return the status page immediately
        return render_template('import_status.html', import_id=import_id, import_type='purchases')
        
    except Exception as e:
        logger.error(f"Error starting import: {str(e)}")
        flash(f'Error starting import: {str(e)}', 'error')
        return redirect(url_for('import_bp.import_purchases'))

# Status check route
@import_bp.route('/import/status/<import_id>', methods=['GET'])
def check_import_status(import_id):
    status_data = get_import_status(import_id)
    if status_data:
        return jsonify(status_data)
    else:
        return jsonify({'status': 'error', 'message': 'Import not found'})

