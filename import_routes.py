import os
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app
from werkzeug.utils import secure_filename
from datetime import datetime
import pytz
import re
from models import db, Customer, Vendor, Product, Sale, SaleItem, Purchase, PurchaseItem

# Create a Blueprint for import routes
import_bp = Blueprint('import_bp', __name__)

# Define IST timezone
IST = pytz.timezone('Asia/Kolkata')

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['xlsx', 'xls', 'csv']

# Helper function to create upload directory if it doesn't exist
def ensure_upload_dir():
    upload_dir = os.path.join(current_app.root_path, 'uploads')
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    return upload_dir

# Helper function to parse date based on format
def parse_date(date_str, date_format):
    try:
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
            
            # File is valid, proceed to date format selection
            return render_template('import_sales.html', step=2, file_path=file_path)
            
        except Exception as e:
            flash(f'Error validating file: {str(e)}', 'error')
            if os.path.exists(file_path):
                os.remove(file_path)
            return redirect(url_for('import_bp.import_sales'))
    else:
        flash('Invalid file type. Please upload an Excel (.xlsx, .xls) or CSV file.', 'error')
        return redirect(url_for('import_bp.import_sales'))

@import_bp.route('/import/sales/preview', methods=['POST'])
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
        
        # Check for date parsing issues
        date_warnings = False
        parsed_dates = []
        
        for date_str in df['Date']:
            try:
                parsed_date = parse_date(str(date_str), date_format)
                parsed_dates.append(parsed_date)
            except ValueError:
                date_warnings = True
                parsed_dates.append(None)
        
        df['parsed_date'] = parsed_dates
        
        # Prepare preview data
        preview_data = []
        
        for _, row in df.head(10).iterrows():
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
        
        # Check if products exist
        product_names = df['Product Name'].unique()
        existing_products = Product.query.filter(Product.name.in_(product_names)).all()
        existing_product_names = [p.name for p in existing_products]
        missing_products = [name for name in product_names if name not in existing_product_names]
        
        # Check if customers exist
        customer_names = df['Customer Name'].unique()
        existing_customers = Customer.query.filter(Customer.name.in_(customer_names)).all()
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
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('import_bp.import_sales'))

@import_bp.route('/import/sales/process', methods=['POST'])
def process_sales_import():
    file_path = request.form.get('file_path')
    date_format = request.form.get('date_format')
    create_missing = request.form.get('create_missing') == 'yes'
    
    if not file_path or not os.path.exists(file_path):
        flash('File not found. Please upload again.', 'error')
        return redirect(url_for('import_bp.import_sales'))
    
    try:
        # Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Parse dates
        parsed_dates = []
        for date_str in df['Date']:
            try:
                parsed_date = parse_date(str(date_str), date_format)
                parsed_dates.append(parsed_date)
            except ValueError:
                parsed_dates.append(None)
        
        df['parsed_date'] = parsed_dates
        
        # Initialize results
        import_results = {
            'success': 0,
            'warnings': 0,
            'errors': 0,
            'new_customers': [],
            'new_products': [],
            'errors_list': []
        }
        
        # Get existing products and customers
        all_products = {p.name: p for p in Product.query.all()}
        all_customers = {c.name: c for c in Customer.query.all()}
        
        # Group by invoice number and date
        grouped = df.groupby(['Invoice Number', 'Customer Name'])
        
        for (invoice_number, customer_name), group_data in grouped:
            try:
                # Get or create customer
                if customer_name in all_customers:
                    customer = all_customers[customer_name]
                elif create_missing:
                    # Create a new customer with placeholder data
                    customer = Customer(
                        name=customer_name,
                        gst_id=f"CUST{len(all_customers) + 1}",  # Placeholder GST ID
                        location="Unknown",  # Placeholder location
                        contact_person="",
                        phone="",
                        about=f"Imported from Excel on {datetime.now().strftime('%Y-%m-%d')}"
                    )
                    db.session.add(customer)
                    db.session.flush()  # Get ID without committing
                    all_customers[customer_name] = customer
                    import_results['new_customers'].append(customer_name)
                else:
                    import_results['errors'] += 1
                    import_results['errors_list'].append(f"Customer not found: {customer_name}")
                    continue
                
                # Get the first valid date from the group
                valid_dates = [d for d in group_data['parsed_date'] if pd.notna(d)]
                if valid_dates:
                    sale_date = valid_dates[0]
                else:
                    sale_date = datetime.now()
                
                # Calculate total amount for the sale
                total_amount = 0
                delivery_charges = 0  # Default to 0, can be updated if in the data
                
                # Create sale
                sale = Sale(
                    customer_id=customer.id,
                    sale_date=sale_date,
                    delivery_charges=delivery_charges,
                    total_amount=0  # Will update after calculating items
                )
                db.session.add(sale)
                db.session.flush()  # Get ID without committing
                
                # Process each item in the sale
                for _, row in group_data.iterrows():
                    product_name = row['Product Name']
                    
                    # Get or create product
                    if product_name in all_products:
                        product = all_products[product_name]
                    elif create_missing:
                        # Generate a unique product ID
                        product_id = generate_product_id(product_name)
                        
                        # Create a new product with placeholder data
                        product = Product(
                            product_id=product_id,
                            name=product_name,
                            quantity=0,  # Initial quantity
                            cost_per_unit=0,  # Will be updated based on sales/purchases
                            specifications=""
                        )
                        db.session.add(product)
                        db.session.flush()  # Get ID without committing
                        all_products[product_name] = product
                        import_results['new_products'].append(product_name)
                    else:
                        import_results['errors'] += 1
                        import_results['errors_list'].append(f"Product not found: {product_name}")
                        continue
                    
                    # Calculate item price
                    quantity = int(row['Quantity'])
                    total_amount_item = float(row['Total Amount'])
                    gst_percentage = 18.0  # Default GST
                    discount_percentage = 0.0  # Default discount
                    
                    # Calculate pre-tax amount
                    pre_tax_amount = total_amount_item / (1 + gst_percentage/100)
                    unit_price = pre_tax_amount / quantity
                    
                    # Check if enough stock
                    if product.quantity < quantity:
                        import_results['warnings'] += 1
                        import_results['errors_list'].append(f"Warning: Not enough stock for product {product_name}. Current: {product.quantity}, Required: {quantity}")
                    
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
                    
                    # Add to total amount
                    total_amount += total_amount_item
                
                # Update sale total amount
                sale.total_amount = total_amount
                
                import_results['success'] += 1
                
            except Exception as e:
                import_results['errors'] += 1
                import_results['errors_list'].append(f"Error importing sale {invoice_number}: {str(e)}")
        
        # Commit all changes
        db.session.commit()
        
        # Clean up the file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return render_template('import_sales.html', step=4, import_results=import_results)
    
    except Exception as e:
        db.session.rollback()
        flash(f'Error importing data: {str(e)}', 'error')
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
            except ValueError:
                date_warnings = True
                parsed_dates.append(None)
        
        df['parsed_date'] = parsed_dates
        
        # Prepare preview data
        preview_data = []
        
        for _, row in df.head(10).iterrows():
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
        
        # Check if products exist
        product_names = df['Product Name'].unique()
        existing_products = Product.query.filter(Product.name.in_(product_names)).all()
        existing_product_names = [p.name for p in existing_products]
        missing_products = [name for name in product_names if name not in existing_product_names]
        
        # Check if vendors exist
        vendor_names = df['Vendor Name'].unique()
        existing_vendors = Vendor.query.all()
        existing_vendor_names = [v.name.strip().lower() for v in existing_vendors]
        missing_vendors = [name for name in vendor_names if name.strip().lower() not in existing_vendor_names]
        
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

@import_bp.route('/import/purchases/process', methods=['POST'])
def process_purchases_import():
    file_path = request.form.get('file_path')
    date_format = request.form.get('date_format')
    create_missing_products = request.form.get('create_missing') == 'yes'
    
    if not file_path or not os.path.exists(file_path):
        flash('File not found. Please upload again.', 'error')
        return redirect(url_for('import_bp.import_purchases'))
    
    try:
        # Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Parse dates
        parsed_dates = []
        for date_str in df['Date']:
            try:
                parsed_date = parse_date(str(date_str), date_format)
                parsed_dates.append(parsed_date)
            except ValueError:
                parsed_dates.append(None)
        
        df['parsed_date'] = parsed_dates
        
        # Initialize results
        import_results = {
            'success': 0,
            'warnings': 0,
            'errors': 0,
            'new_vendors': [],
            'new_products': [],
            'errors_list': []
        }
        
        # Get existing products
        all_products = {p.name: p for p in Product.query.all()}
        
        # Get all vendors and create a case-insensitive lookup dictionary
        existing_vendors = Vendor.query.all()
        all_vendors = {}
        for v in existing_vendors:
            all_vendors[v.name] = v
            all_vendors[v.name.strip().lower()] = v
        
        # Pre-create all vendors from the import file that don't exist yet
        vendor_names = df['Vendor Name'].unique()
        for vendor_name in vendor_names:
            vendor_key = vendor_name.strip().lower()
            if vendor_name not in all_vendors and vendor_key not in all_vendors:
                # Create a new vendor with data from the import
                vendor = Vendor(
                    name=vendor_name,
                    gst_id=f"VEND{len(all_vendors) // 2 + 1}",  # Generate a unique GST ID
                    location="Unknown",  # Placeholder location
                    contact_person="",
                    phone="",
                    about=f"Auto-created from import on {datetime.now().strftime('%Y-%m-%d')}"
                )
                db.session.add(vendor)
                import_results['new_vendors'].append(vendor_name)
                
                # Add to the vendors dictionary
                all_vendors[vendor_name] = vendor
                all_vendors[vendor_key] = vendor
        
        # Flush to get IDs for the new vendors
        db.session.flush()
        
        # Group by order ID and vendor
        grouped = df.groupby(['Order ID', 'Vendor Name'])
        
        for (order_id, vendor_name), group_data in grouped:
            try:
                # Get the vendor (should always exist now)
                vendor_key = vendor_name.strip().lower()
                if vendor_name in all_vendors:
                    vendor = all_vendors[vendor_name]
                elif vendor_key in all_vendors:
                    vendor = all_vendors[vendor_key]
                else:
                    # This should never happen now, but just in case
                    import_results['errors'] += 1
                    import_results['errors_list'].append(f"Vendor not found: {vendor_name}")
                    continue
                
                # Get the first valid date from the group
                valid_dates = [d for d in group_data['parsed_date'] if pd.notna(d)]
                if valid_dates:
                    purchase_date = valid_dates[0]
                else:
                    purchase_date = datetime.now()
                
                # Set default status to Delivered
                status = "Delivered"
                
                # Calculate total amount for the purchase
                total_amount = 0
                delivery_charges = 0  # Default to 0, can be updated if in the data
                
                # Create purchase
                purchase = Purchase(
                    vendor_id=vendor.id,
                    purchase_date=purchase_date,
                    order_id=order_id,
                    delivery_charges=delivery_charges,
                    total_amount=0,  # Will update after calculating items
                    status=status
                )
                db.session.add(purchase)
                db.session.flush()  # Get ID without committing
                
                # Process each item in the purchase
                for _, row in group_data.iterrows():
                    product_name = row['Product Name']
                    
                    # Get or create product
                    if product_name in all_products:
                        product = all_products[product_name]
                    elif create_missing_products:
                        # Generate a unique product ID
                        product_id = generate_product_id(product_name)
                        
                        # Create a new product with placeholder data
                        product = Product(
                            product_id=product_id,
                            name=product_name,
                            quantity=0,  # Initial quantity
                            cost_per_unit=0,  # Will be updated based on purchases
                            specifications=""
                        )
                        db.session.add(product)
                        db.session.flush()  # Get ID without committing
                        all_products[product_name] = product
                        import_results['new_products'].append(product_name)
                    else:
                        import_results['errors'] += 1
                        import_results['errors_list'].append(f"Product not found: {product_name}")
                        continue
                    
                    # Calculate item price
                    quantity = int(row['Quantity'])
                    total_amount_item = float(row['Total Amount'])
                    gst_percentage = 18.0  # Default GST
                    
                    # Calculate pre-tax amount
                    pre_tax_amount = total_amount_item / (1 + gst_percentage/100)
                    unit_price = pre_tax_amount / quantity
                    
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
                    
                    # Update product quantity and cost
                    old_quantity = product.quantity
                    product.quantity += quantity
                    
                    # Update cost_per_unit as weighted average
                    if product.quantity > 0:
                        if old_quantity > 0 and product.cost_per_unit > 0:
                            product.cost_per_unit = ((product.cost_per_unit * old_quantity) + (unit_price * quantity)) / product.quantity
                        else:
                            product.cost_per_unit = unit_price
                    
                    # Add to total amount
                    total_amount += total_amount_item
                
                # Update purchase total amount
                purchase.total_amount = total_amount
                
                import_results['success'] += 1
                
            except Exception as e:
                import_results['errors'] += 1
                import_results['errors_list'].append(f"Error importing purchase {order_id}: {str(e)}")
        
        # Commit all changes
        db.session.commit()
        
        # Clean up the file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return render_template('import_purchases.html', step=4, import_results=import_results)
    
    except Exception as e:
        db.session.rollback()
        flash(f'Error importing data: {str(e)}', 'error')
        return redirect(url_for('import_bp.import_purchases'))

