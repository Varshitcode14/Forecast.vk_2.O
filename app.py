from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, abort, g
from flask_sqlalchemy import SQLAlchemy
from models import db, User, Customer, Product, Sale, SaleItem, Vendor, Purchase, PurchaseItem, Report
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import os
import json
from decimal import Decimal
from datetime import datetime, timedelta
import pytz
from import_routes import import_bp
from functools import wraps
import time
import logging
from sqlalchemy import or_
import pandas as pd
import numpy as np

# Configure logging FIRST - before any other code
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
    logger.info("ARIMA library loaded successfully")
except ImportError:
    ARIMA_AVAILABLE = False
    logger.warning("ARIMA not available. Install statsmodels for advanced forecasting.")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'production-key-for-deployment')

# Direct database configuration with the new URL
database_url = "postgresql://forecast_may_user:iDYGvjoHXOBI9JnAl8v3Xn4l5G9yywm6@dpg-d0fd0qruibrs73eipseg-a.oregon-postgres.render.com/forecast_may"
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,
    'max_overflow': 15,
    'pool_timeout': 30,
    'pool_recycle': 60,
    'pool_pre_ping': True,
}

# PostgreSQL specific connect args
app.config['SQLALCHEMY_ENGINE_OPTIONS']['connect_args'] = {
    'connect_timeout': 10,
    'application_name': 'forecast_vk_app',
    'keepalives': 1,
    'keepalives_idle': 30,
    'keepalives_interval': 10,
    'keepalives_count': 5
}

# Log the database URL being used (with password masked for security)
masked_url = database_url.replace(database_url.split('@')[0].split(':', 2)[2], '********')
print(f"Using database URL: {masked_url}")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db.init_app(app)

# Add a function to test database connectivity
def test_db_connection():
    try:
        # Try to execute a simple query
        with app.app_context():
            db.session.execute('SELECT 1')
            logger.info("Database connection successful")
            return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False

# Test the database connection at startup
with app.app_context():
    connection_successful = test_db_connection()
    if not connection_successful:
        logger.critical("Failed to connect to database. Check connection parameters and network.")

# Add this to your app.py to ensure database connections are properly handled in threads
from sqlalchemy import event
from sqlalchemy.engine import Engine
import threading

# Ensure each thread gets its own database connection
@app.before_request
def before_request():
    if not hasattr(g, 'db_conn'):
        g.db_conn = db.engine.connect()

@app.teardown_request
def teardown_request(exception):
    if hasattr(g, 'db_conn'):
        g.db_conn.close()

# Define IST timezone
IST = pytz.timezone('Asia/Kolkata')

# Helper function to get current time in IST
def get_current_time_ist():
    return datetime.now(pytz.utc).astimezone(IST)

# Function to create admin user with better error handling
def create_admin_user():
    try:
        # Check if admin user exists
        admin_email = "forecastai007@gmail.com"
        admin = User.query.filter_by(email=admin_email).first()
        
        if not admin:
            # Create admin user
            admin = User(
                name="admin_Varshit_k",
                email=admin_email,
                is_admin=True
            )
            admin.set_password("Forecast@007")
            db.session.add(admin)
            db.session.commit()
            logger.info("Admin user created successfully")
            return True
        elif not admin.is_admin:
            # Update existing user to be admin
            admin.is_admin = True
            db.session.commit()
            logger.info("User updated to admin status")
            return True
        else:
            logger.info("Admin user already exists")
            return True
    except Exception as e:
        logger.error(f"Error creating admin user: {str(e)}")
        db.session.rollback()
        return False

def arima_forecast(sales_data, steps=3):
    """
    Generate ARIMA forecast for sales data
    """
    try:
        if not ARIMA_AVAILABLE:
            return None
            
        # Convert to pandas series
        if len(sales_data) < 10:  # Need minimum data for ARIMA
            return None
            
        # Remove zeros and handle missing data
        sales_series = pd.Series(sales_data)
        sales_series = sales_series.replace(0, np.nan).fillna(method='ffill').fillna(method='bfill')
        
        if sales_series.isna().all() or len(sales_series.dropna()) < 5:
            return None
            
        # Try different ARIMA orders and pick the best one
        best_aic = float('inf')
        best_forecast = None
        
        # Simple grid search for best parameters
        for p in [0, 1, 2]:
            for d in [0, 1]:
                for q in [0, 1, 2]:
                    try:
                        model = ARIMA(sales_series.dropna(), order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            forecast_result = fitted_model.forecast(steps=steps)
                            best_forecast = forecast_result.tolist() if hasattr(forecast_result, 'tolist') else list(forecast_result)
                    except:
                        continue
        
        return best_forecast
        
    except Exception as e:
        logger.error(f"ARIMA forecasting error: {str(e)}")
        return None

# Create tables and initialize admin user with better error handling
with app.app_context():
    try:
        db.create_all()
        logger.info("Database tables created successfully")
        
        # Create admin user and log the result
        admin_created = create_admin_user()
        if admin_created:
            logger.info("Admin user setup completed successfully")
        else:
            logger.warning("Admin user setup failed")
            
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")

# Register the import blueprint
app.register_blueprint(import_bp)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception as e:
        logger.error(f"Error loading user: {str(e)}")
        return None

# Add this decorator function to check if user is admin
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You do not have permission to access this page', 'error')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

# Add this decorator function to check if user is NOT an admin
def regular_user_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        if current_user.is_admin:
            flash('Admin users should use the admin dashboard', 'error')
            return redirect(url_for('admin_dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Authentication routes
@app.route('/')
def landing():
    return render_template('landing.html')

# Update the signup route with better error handling

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            # Log the signup attempt
            logger.info(f"Signup attempt for email: {email}")
            
            # Validate form data
            if not name or not email or not password or not confirm_password:
                flash('All fields are required', 'error')
                return redirect(url_for('signup'))
                
            if password != confirm_password:
                flash('Passwords do not match', 'error')
                return redirect(url_for('signup'))
                
            # Check if user already exists
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash('Email already registered', 'error')
                return redirect(url_for('signup'))
                
            # Create new user
            new_user = User(name=name, email=email)
            new_user.set_password(password)
            
            db.session.add(new_user)
            db.session.commit()
            
            logger.info(f"User created successfully: {email}")
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error in signup: {str(e)}")
            flash('An error occurred during signup. Please try again.', 'error')
            return redirect(url_for('signup'))
            
    return render_template('signup.html')

# Update the login route with better error handling and debugging

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            password = request.form.get('password')
            
            logger.info(f"Login attempt for email: {email}")
            
            user = User.query.filter_by(email=email).first()
            
            if not user:
                logger.warning(f"Login failed: No user found with email {email}")
                flash('Invalid email or password', 'error')
                return render_template('login.html')
                
            if not user.check_password(password):
                logger.warning(f"Login failed: Incorrect password for {email}")
                flash('Invalid email or password', 'error')
                return render_template('login.html')
            
            # If we get here, login is successful
            login_user(user)
            
            # Log successful login
            logger.info(f"User logged in successfully: {email}, is_admin: {user.is_admin}")
            
            # Redirect admin users to admin dashboard
            if user.is_admin:
                flash('Admin logged in successfully!', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                flash('Logged in successfully!', 'success')
                return redirect(url_for('home'))
                
        except Exception as e:
            logger.error(f"Error in login: {str(e)}")
            flash('An error occurred during login. Please try again.', 'error')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('landing'))

@app.route('/home')
@login_required
@regular_user_required
def home():
    return render_template('index.html')

# Function to generate daily report
def generate_daily_report():
    with app.app_context():
        try:
            # Get today's date in IST
            today = get_current_time_ist().date()
            
            # Get all users
            users = User.query.all()
            
            for user in users:
                # Get sales and purchases for today for this user
                sales = Sale.query.filter(
                    Sale.user_id == user.id,
                    Sale.sale_date >= today,
                    Sale.sale_date < today + timedelta(days=1)
                ).all()
                
                purchases = Purchase.query.filter(
                    Purchase.user_id == user.id,
                    Purchase.purchase_date >= today,
                    Purchase.purchase_date < today + timedelta(days=1)
                ).all()
                
                # Calculate totals
                total_sales = sum(sale.total_amount for sale in sales)
                total_purchases = sum(purchase.total_amount for purchase in purchases)
                net_profit = total_sales - total_purchases
                profit_margin = (net_profit / total_sales * 100) if total_sales > 0 else 0
                
                # Check inventory status
                products = Product.query.filter_by(user_id=user.id).all()
                low_stock_count = sum(1 for p in products if p.quantity > 0 and p.quantity <= 10)
                out_of_stock_count = sum(1 for p in products if p.quantity <= 0)
                
                # Create a new report object
                report = Report(
                    user_id=user.id,
                    report_type='daily',
                    start_date=today,
                    end_date=today,
                    total_sales=total_sales,
                    total_purchases=total_purchases,
                    net_profit=net_profit,
                    profit_margin=profit_margin,
                    low_stock_count=low_stock_count,
                    out_of_stock_count=out_of_stock_count
                )
                
                db.session.add(report)
            
            db.session.commit()
            logger.info(f"Daily Reports Generated for {today}")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error generating daily report: {str(e)}")

@app.route('/insights')
@login_required
@regular_user_required
def insights():
    return render_template('insights.html')

@app.route('/forecast_reports')
@login_required
@regular_user_required
def forecast_reports():
    return render_template('forecast_reports.html')

@app.route('/stock')
@login_required
@regular_user_required
def stock():
    try:
        products = Product.query.filter_by(user_id=current_user.id).all()
        return render_template('stock.html', products=products)
    except Exception as e:
        logger.error(f"Error in stock route: {str(e)}")
        flash('Error loading stock data. Please try again.', 'error')
        return redirect(url_for('home'))

@app.route('/sales')
@login_required
@regular_user_required
def sales():
    try:
        sales = Sale.query.filter_by(user_id=current_user.id).order_by(Sale.sale_date.desc()).all()
        return render_template('sales.html', sales=sales)
    except Exception as e:
        logger.error(f"Error in sales route: {str(e)}")
        flash('Error loading sales data. Please try again.', 'error')
        return redirect(url_for('home'))

@app.route('/purchases')
@login_required
@regular_user_required
def purchases():
    try:
        purchases = Purchase.query.filter_by(user_id=current_user.id).order_by(Purchase.purchase_date.desc()).all()
        return render_template('purchases.html', purchases=purchases)
    except Exception as e:
        logger.error(f"Error in purchases route: {str(e)}")
        flash('Error loading purchases data. Please try again.', 'error')
        return redirect(url_for('home'))

@app.route('/customers')
@login_required
@regular_user_required
def customers():
    try:
        customers = Customer.query.filter_by(user_id=current_user.id).all()
        return render_template('customers.html', customers=customers)
    except Exception as e:
        logger.error(f"Error in customers route: {str(e)}")
        flash('Error loading customer data. Please try again.', 'error')
        return redirect(url_for('home'))

@app.route('/vendors')
@login_required
@regular_user_required
def vendors():
    try:
        vendors = Vendor.query.filter_by(user_id=current_user.id).all()
        return render_template('vendors.html', vendors=vendors)
    except Exception as e:
        logger.error(f"Error in vendors route: {str(e)}")
        flash('Error loading vendor data. Please try again.', 'error')
        return redirect(url_for('home'))

@app.route('/notifications')
@login_required
@regular_user_required
def notifications():
    return render_template('notifications.html')

@app.route('/report')
@login_required
@regular_user_required
def report():
    return render_template('report.html')

# Import data route
@app.route('/import_data')
@login_required
@regular_user_required
def import_data():
    return redirect(url_for('import_bp.import_data'))

# Customer API endpoints
@app.route('/api/customers', methods=['GET'])
@login_required
def get_customers():
    try:
        customers = Customer.query.filter_by(user_id=current_user.id).all()
        return jsonify([customer.to_dict() for customer in customers])
    except Exception as e:
        logger.error(f"Error in get_customers: {str(e)}")
        return jsonify({'error': 'Failed to fetch customers'}), 500

@app.route('/api/customers', methods=['POST'])
@login_required
def add_customer():
    try:
        data = request.json

        # Check if customer with GST ID already exists for this user
        existing_customer = Customer.query.filter_by(user_id=current_user.id, gst_id=data['gst_id']).first()
        if existing_customer:
            return jsonify({'success': False, 'message': 'Customer with this GST ID already exists'}), 400

        customer = Customer(
            user_id=current_user.id,
            gst_id=data['gst_id'],
            name=data['name'],
            contact_person=data.get('contact_person', ''),
            phone=data.get('phone', ''),
            location=data['location'],
            about=data.get('about', '')
        )

        db.session.add(customer)
        db.session.commit()

        return jsonify({'success': True, 'customer': customer.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in add_customer: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while adding the customer'}), 500

@app.route('/api/customers/<int:id>', methods=['PUT'])
@login_required
def update_customer(id):
    try:
        customer = Customer.query.get_or_404(id)
        data = request.json

        customer.name = data.get('name', customer.name)
        customer.contact_person = data.get('contact_person', customer.contact_person)
        customer.phone = data.get('phone', customer.phone)
        customer.location = data.get('location', customer.location)
        customer.about = data.get('about', customer.about)

        db.session.commit()

        return jsonify({'success': True, 'customer': customer.to_dict()})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in update_customer: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while updating the customer'}), 500

@app.route('/api/customers/<int:id>', methods=['DELETE'])
@login_required
def delete_customer(id):
    try:
        customer = Customer.query.get_or_404(id)

        # Check if customer has sales
        if customer.sales:
            return jsonify({'success': False, 'message': 'Cannot delete customer with sales records'}), 400

        db.session.delete(customer)
        db.session.commit()

        return jsonify({'success': True}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in delete_customer: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while deleting the customer'}), 500

# Vendor API endpoints
@app.route('/api/vendors', methods=['GET'])
@login_required
def get_vendors():
    try:
        vendors = Vendor.query.filter_by(user_id=current_user.id).all()
        return jsonify([vendor.to_dict() for vendor in vendors])
    except Exception as e:
        logger.error(f"Error in get_vendors: {str(e)}")
        return jsonify({'error': 'Failed to fetch vendors'}), 500

@app.route('/api/vendors', methods=['POST'])
@login_required
def add_vendor():
    try:
        data = request.json

        # Check if vendor with GST ID already exists for this user
        existing_vendor = Vendor.query.filter_by(user_id=current_user.id, gst_id=data['gst_id']).first()
        if existing_vendor:
            return jsonify({'success': False, 'message': 'Vendor with this GST ID already exists'}), 400

        vendor = Vendor(
            user_id=current_user.id,
            gst_id=data['gst_id'],
            name=data['name'],
            contact_person=data.get('contact_person', ''),
            phone=data.get('phone', ''),
            location=data['location'],
            about=data.get('about', '')
        )

        db.session.add(vendor)
        db.session.commit()

        return jsonify({'success': True, 'vendor': vendor.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in add_vendor: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while adding the vendor'}), 500

@app.route('/api/vendors/<int:id>', methods=['PUT'])
@login_required
def update_vendor(id):
    try:
        vendor = Vendor.query.get_or_404(id)
        data = request.json

        vendor.name = data.get('name', vendor.name)
        vendor.contact_person = data.get('contact_person', vendor.contact_person)
        vendor.phone = data.get('phone', vendor.phone)
        vendor.location = data.get('location', vendor.location)
        vendor.about = data.get('about', vendor.about)

        db.session.commit()

        return jsonify({'success': True, 'vendor': vendor.to_dict()})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in update_vendor: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while updating the vendor'}), 500

@app.route('/api/vendors/<int:id>', methods=['DELETE'])
@login_required
def delete_vendor(id):
    try:
        vendor = Vendor.query.get_or_404(id)

        # Check if vendor has purchases
        if vendor.purchases:
            return jsonify({'success': False, 'message': 'Cannot delete vendor with purchase records'}), 400

        db.session.delete(vendor)
        db.session.commit()

        return jsonify({'success': True}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in delete_vendor: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while deleting the vendor'}), 500

# Product API endpoints
@app.route('/api/products', methods=['GET'])
@login_required
def get_products():
    try:
        products = Product.query.filter_by(user_id=current_user.id).all()
        return jsonify([product.to_dict() for product in products])
    except Exception as e:
        logger.error(f"Error in get_products: {str(e)}")
        return jsonify({'error': 'Failed to fetch products'}), 500

@app.route('/api/products', methods=['POST'])
@login_required
def add_product():
    try:
        data = request.json

        # Check if product with product ID already exists for this user
        existing_product = Product.query.filter_by(user_id=current_user.id, product_id=data['product_id']).first()
        if existing_product:
            return jsonify({'success': False, 'message': 'Product with this ID already exists'}), 400

        product = Product(
            user_id=current_user.id,
            product_id=data['product_id'],
            name=data['name'],
            quantity=data['quantity'],
            cost_per_unit=data['cost_per_unit'],
            specifications=data.get('specifications', '')
        )

        db.session.add(product)
        db.session.commit()

        return jsonify({'success': True, 'product': product.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in add_product: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while adding the product'}), 500

@app.route('/api/products/<int:id>', methods=['PUT'])
@login_required
def update_product(id):
    try:
        product = Product.query.get_or_404(id)
        data = request.json

        product.name = data.get('name', product.name)
        product.quantity = data.get('quantity', product.quantity)
        product.cost_per_unit = data.get('cost_per_unit', product.cost_per_unit)
        product.specifications = data.get('specifications', product.specifications)

        db.session.commit()

        return jsonify({'success': True, 'product': product.to_dict()})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in update_product: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while updating the product'}), 500

@app.route('/api/products/<int:id>', methods=['DELETE'])
@login_required
def delete_product(id):
    try:
        product = Product.query.get_or_404(id)

        # Check if product has sale items or purchase items
        if product.sale_items or product.purchase_items:
            return jsonify({'success': False, 'message': 'Cannot delete product with sales or purchase records'}), 400

        db.session.delete(product)
        db.session.commit()

        return jsonify({'success': True}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in delete_product: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while deleting the product'}), 500

# Sales API endpoints
@app.route('/api/sales', methods=['POST'])
@login_required
def add_sale():
    try:
        data = request.json

        # Get customer
        customer = Customer.query.filter_by(id=data['customer_id'], user_id=current_user.id).first()
        if not customer:
            return jsonify({'success': False, 'message': 'Customer not found'}), 404

        # Create sale with IST time
        sale = Sale(
            user_id=current_user.id,
            customer_id=customer.id,
            delivery_charges=data['delivery_charges'],
            total_amount=data['total_amount'],
            sale_date=get_current_time_ist()
        )

        db.session.add(sale)

        # Add sale items
        for item_data in data['items']:
            product = Product.query.filter_by(id=item_data['product_id'], user_id=current_user.id).first()
            if not product:
                db.session.rollback()
                return jsonify({'success': False, 'message': f'Product not found: {item_data["product_id"]}'}), 404
          
            # Check if enough stock
            if product.quantity < item_data['quantity']:
                db.session.rollback()
                return jsonify({'success': False, 'message': f'Not enough stock for product: {product.name}'}), 400
          
            # Update product quantity
            product.quantity -= item_data['quantity']
          
            # Create sale item
            sale_item = SaleItem(
                sale=sale,
                product_id=product.id,
                quantity=item_data['quantity'],
                gst_percentage=item_data['gst_percentage'],
                discount_percentage=item_data['discount_percentage'],
                unit_price=item_data.get('unit_price', product.cost_per_unit),  # Use provided unit price or default to product cost
                total_price=item_data['total_price']
            )
          
            db.session.add(sale_item)

        db.session.commit()

        return jsonify({'success': True, 'sale_id': sale.id}), 201
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in add_sale: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while adding the sale'}), 500

@app.route('/api/sales', methods=['GET'])
@login_required
def get_sales():
  try:
      # Get pagination parameters
      page = request.args.get('page', 1, type=int)
      per_page = request.args.get('per_page', 1, type=int)
      
      # Limit per_page to prevent excessive data loads
      per_page = min(per_page, 100)
      
      # Get search parameter
      search_term = request.args.get('search', '')
      
      # Log request parameters
      logger.info(f"Fetching sales: page={page}, per_page={per_page}, search={search_term}")
      
      # Build the base query with eager loading to reduce database hits
      query = Sale.query.filter_by(user_id=current_user.id)
      
      # Add search filter if provided
      if search_term:
          # Join with Customer to search by customer name
          query = query.join(Customer).filter(
              or_(
                  Customer.name.ilike(f"%{search_term}%"),
                  Customer.gst_id.ilike(f"%{search_term}%")
              )
          )
      
      # Order by sale date descending (newest first)
      query = query.order_by(Sale.sale_date.desc())
      
      # Execute the paginated query
      paginated_sales = query.paginate(page=page, per_page=per_page, error_out=False)
      
      # Get the total count
      total_count = paginated_sales.total
      logger.info(f"Found {total_count} total sales, returning page {page} with {len(paginated_sales.items)} items")
      
      # Process the results
      result = []
      for sale in paginated_sales.items:
          try:
              # Format date in IST - handle timezone-aware and naive datetime objects
              if sale.sale_date:
                  if sale.sale_date.tzinfo:
                      sale_date_ist = sale.sale_date.astimezone(IST)
                  else:
                      sale_date_ist = IST.localize(sale.sale_date)
                  date_str = sale_date_ist.strftime('%Y-%m-%d')
              else:
                  date_str = "N/A"
              
              # Get customer details safely
              customer_name = sale.customer.name if sale.customer else "Unknown"
              customer_gst_id = sale.customer.gst_id if sale.customer else "Unknown"
              
              sale_data = {
                  'id': sale.id,
                  'customer_name': customer_name,
                  'customer_gst_id': customer_gst_id,
                  'sale_date': date_str,
                  'delivery_charges': float(sale.delivery_charges) if sale.delivery_charges else 0.0,
                  'total_amount': float(sale.total_amount) if sale.total_amount else 0.0,
                  'items': []
              }
              
              # Process each item with error handling
              for item in sale.items:
                  try:
                      if not item.product:
                          logger.warning(f"Sale item {item.id} has no associated product")
                          continue
                          
                      item_data = {
                          'product_name': item.product.name,
                          'product_id': item.product.product_id,
                          'quantity': item.quantity,
                          'gst_percentage': float(item.gst_percentage) if item.gst_percentage else 0.0,
                          'discount_percentage': float(item.discount_percentage) if item.discount_percentage else 0.0,
                          'unit_price': float(item.unit_price) if item.unit_price else 0.0,
                          'total_price': float(item.total_price) if item.total_price else 0.0
                      }
                      sale_data['items'].append(item_data)
                  except Exception as item_error:
                      logger.error(f"Error processing sale item {item.id}: {str(item_error)}")
                      continue
              
              result.append(sale_data)
          except Exception as sale_error:
              logger.error(f"Error processing sale {sale.id}: {str(sale_error)}")
              continue

      # Return paginated response
      return jsonify({
          'sales': result,
          'pagination': {
              'page': page,
              'per_page': per_page,
              'total': total_count,
              'pages': paginated_sales.pages
          }
      })
  except Exception as e:
      logger.error(f"Error in get_sales: {str(e)}")
      return jsonify({'error': 'Failed to fetch sales', 'message': str(e)}), 500

@app.route('/api/sales/<int:id>', methods=['DELETE'])
@login_required
def delete_sale(id):
    try:
        sale = Sale.query.get_or_404(id)
        
        # First, restore product quantities
        for item in sale.items:
            product = item.product
            product.quantity += item.quantity
        
        # Delete all sale items
        for item in sale.items:
            db.session.delete(item)
        
        # Delete the sale
        db.session.delete(sale)
        db.session.commit()
        
        return jsonify({'success': True}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in delete_sale: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while deleting the sale'}), 500

# Purchases API endpoints
@app.route('/api/purchases', methods=['POST'])
@login_required
def add_purchase():
    try:
        data = request.json

        # Get vendor
        vendor = Vendor.query.filter_by(id=data['vendor_id'], user_id=current_user.id).first()
        if not vendor:
            return jsonify({'success': False, 'message': 'Vendor not found'}), 404

        # Create purchase with IST time
        purchase = Purchase(
            user_id=current_user.id,
            vendor_id=vendor.id,
            order_id=data['order_id'],
            delivery_charges=data['delivery_charges'],
            total_amount=data['total_amount'],
            status=data['status'],
            purchase_date=get_current_time_ist()
        )

        if 'purchase_date' in data:
            purchase_date = datetime.strptime(data['purchase_date'], '%Y-%m-%d')
            purchase.purchase_date = IST.localize(purchase_date)

        db.session.add(purchase)

        # Add purchase items
        for item_data in data['items']:
            product = Product.query.filter_by(id=item_data['product_id'], user_id=current_user.id).first()
            if not product:
                db.session.rollback()
                return jsonify({'success': False, 'message': f'Product not found: {item_data["product_id"]}'}), 404
          
            # Create purchase item
            purchase_item = PurchaseItem(
                purchase=purchase,
                product_id=product.id,
                quantity=item_data['quantity'],
                gst_percentage=item_data['gst_percentage'],
                unit_price=item_data['unit_price'],
                total_price=item_data['total_price']
            )
          
            db.session.add(purchase_item)
          
            # Update product quantity if purchase is delivered
            if data['status'] == 'Delivered':
                product.quantity += item_data['quantity']

        db.session.commit()

        return jsonify({'success': True, 'purchase_id': purchase.id}), 201
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in add_purchase: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while adding the purchase'}), 500

@app.route('/api/purchases', methods=['GET'])
@login_required
def get_purchases():
    try:
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        # Limit per_page to prevent excessive data loads
        per_page = min(per_page, 100)
        
        # Get search parameter
        search_term = request.args.get('search', '')
        
        # Log request parameters
        logger.info(f"Fetching purchases: page={page}, per_page={per_page}, search={search_term}")
        
        # Build the base query with eager loading to reduce database hits
        query = Purchase.query.filter_by(user_id=current_user.id)
        
        # Add search filter if provided
        if search_term:
            # Join with Vendor to search by vendor name
            query = query.join(Vendor).filter(
                or_(
                    Purchase.order_id.ilike(f"%{search_term}%"),
                    Vendor.name.ilike(f"%{search_term}%")
                )
            )
        
        # Order by purchase date descending (newest first)
        query = query.order_by(Purchase.purchase_date.desc())
        
        # Execute the paginated query
        paginated_purchases = query.paginate(page=page, per_page=per_page, error_out=False)
        
        # Get the total count
        total_count = paginated_purchases.total
        logger.info(f"Found {total_count} total purchases, returning page {page} with {len(paginated_purchases.items)} items")
        
        # Process the results
        result = []
        for purchase in paginated_purchases.items:
            try:
                # Format date in IST - handle timezone-aware and naive datetime objects
                if purchase.purchase_date:
                    if purchase.purchase_date.tzinfo:
                        purchase_date_ist = purchase.purchase_date.astimezone(IST)
                    else:
                        purchase_date_ist = IST.localize(purchase.purchase_date)
                    date_str = purchase_date_ist.strftime('%Y-%m-%d')
                else:
                    date_str = "N/A"
                
                # Get vendor details safely
                vendor_name = purchase.vendor.name if purchase.vendor else "Unknown"
                vendor_gst_id = purchase.vendor.gst_id if purchase.vendor else "Unknown"
                
                purchase_data = {
                    'id': purchase.id,
                    'vendor_name': vendor_name,
                    'vendor_gst_id': vendor_gst_id,
                    'order_id': purchase.order_id or "N/A",
                    'purchase_date': date_str,
                    'delivery_charges': float(purchase.delivery_charges) if purchase.delivery_charges else 0.0,
                    'total_amount': float(purchase.total_amount) if purchase.total_amount else 0.0,
                    'status': purchase.status or "Unknown",
                    'items': []
                }
                
                # Process each item with error handling
                for item in purchase.items:
                    try:
                        if not item.product:
                            logger.warning(f"Purchase item {item.id} has no associated product")
                            continue
                            
                        item_data = {
                            'product_name': item.product.name,
                            'product_id': item.product.product_id,
                            'quantity': item.quantity,
                            'gst_percentage': float(item.gst_percentage) if item.gst_percentage else 0.0,
                            'unit_price': float(item.unit_price) if item.unit_price else 0.0,
                            'total_price': float(item.total_price) if item.total_price else 0.0
                        }
                        purchase_data['items'].append(item_data)
                    except Exception as item_error:
                        logger.error(f"Error processing purchase item {item.id}: {str(item_error)}")
                        continue
                
                result.append(purchase_data)
            except Exception as purchase_error:
                logger.error(f"Error processing purchase {purchase.id}: {str(purchase_error)}")
                continue

        # Return paginated response
        return jsonify({
            'purchases': result,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total_count,
                'pages': paginated_purchases.pages
            }
        })
    except Exception as e:
        logger.error(f"Error in get_purchases: {str(e)}")
        return jsonify({'error': 'Failed to fetch purchases', 'message': str(e)}), 500

@app.route('/api/purchases/<int:id>', methods=['PUT'])
@login_required
def update_purchase_status(id):
    try:
        purchase = Purchase.query.get_or_404(id)
        data = request.json

        old_status = purchase.status
        new_status = data.get('status', old_status)

        # If status is changing to Delivered, update product quantities
        if old_status != 'Delivered' and new_status == 'Delivered':
            for item in purchase.items:
                item.product.quantity += item.quantity

        # If status is changing from Delivered to something else, reduce product quantities
        if old_status == 'Delivered' and new_status != 'Delivered':
            for item in purchase.items:
                if item.product.quantity < item.quantity:
                    return jsonify({'success': False, 'message': f'Not enough stock for product: {item.product.name}'}), 400
                item.product.quantity -= item.quantity

        purchase.status = new_status
        db.session.commit()

        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in update_purchase_status: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while updating the purchase status'}), 500

@app.route('/api/purchases/<int:id>', methods=['DELETE'])
@login_required
def delete_purchase(id):
    try:
        purchase = Purchase.query.get_or_404(id)
        
        # If purchase is delivered, reduce product quantities
        if purchase.status == 'Delivered':
            for item in purchase.items:
                product = item.product
                if product.quantity < item.quantity:
                    return jsonify({'success': False, 'message': f'Cannot delete: Product {product.name} has insufficient quantity'}), 400
                product.quantity -= item.quantity
        
        # Delete all purchase items
        for item in purchase.items:
            db.session.delete(item)
        
        # Delete the purchase
        db.session.delete(purchase)
        db.session.commit()
        
        return jsonify({'success': True}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in delete_purchase: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while deleting the purchase'}), 500

# API endpoint for reports
@app.route('/api/reports', methods=['GET'])
@login_required
def get_reports():
  try:
      # Fetch reports from the database for the current user
      reports = Report.query.filter_by(user_id=current_user.id).order_by(Report.created_at.desc()).all()
      report_data = [report.to_dict() for report in reports]
      logger.info(f"Retrieved {len(report_data)} reports for user {current_user.id}")
      return jsonify(report_data)
  except Exception as e:
      logger.error(f"Error in get_reports: {str(e)}")
      return jsonify({'error': 'Failed to fetch reports', 'message': str(e)}), 500

@app.route('/api/reports', methods=['POST'])
@login_required
def add_report():
    try:
        data = request.json
        # In a real application, this would save the report to the database
        return jsonify({'success': True, 'report_id': 4})
    except Exception as e:
        logger.error(f"Error in add_report: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while adding the report'}), 500

@app.route('/api/reports/<int:id>', methods=['GET'])
@login_required
def get_report(id):
    try:
        report = Report.query.get_or_404(id)
        return jsonify(report.to_dict())
    except Exception as e:
        logger.error(f"Error in get_report: {str(e)}")
        return jsonify({'error': 'Failed to fetch report'}), 500

@app.route('/api/arima_forecast', methods=['POST'])
@login_required
def get_arima_forecast():
    """
    Generate ARIMA forecast for given sales data
    """
    try:
        data = request.json
        sales_data = data.get('sales_data', [])
        steps = data.get('steps', 3)
        product_name = data.get('product_name', 'Unknown')
        
        logger.info(f"ARIMA forecast request for {product_name} with {len(sales_data)} data points")
        
        # Generate ARIMA forecast
        arima_result = arima_forecast(sales_data, steps)
        
        if arima_result is None:
            # Fallback to simple moving average
            if len(sales_data) == 0:
                fallback_forecast = [0] * steps
            else:
                avg = sum(sales_data[-3:]) / min(3, len(sales_data))
                fallback_forecast = [max(0, avg)] * steps
            
            return jsonify({
                'success': True,
                'forecast': fallback_forecast,
                'method': 'moving_average',
                'message': 'Used moving average due to insufficient data for ARIMA'
            })
        
        # Ensure non-negative forecasts
        arima_result = [max(0, x) for x in arima_result]
        
        return jsonify({
            'success': True,
            'forecast': arima_result,
            'method': 'arima',
            'message': 'ARIMA forecast generated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in ARIMA forecast API: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'method': 'error'
        }), 500

# Setup scheduler for daily reports
try:
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler()
    scheduler.add_job(generate_daily_report, 'cron', hour=18, minute=0)
    scheduler.start()

    # Register a function to shut down the scheduler when the app is shutting down
    import atexit
    atexit.register(lambda: scheduler.shutdown())

except ImportError:
    logger.warning("APScheduler not installed. Daily reports will not be generated automatically.")

# Admin routes
@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    return render_template('admin.html')

@app.route('/api/admin/users')
@login_required
@admin_required
def get_admin_users():
    try:
        users = User.query.all()
        return jsonify([user.to_dict() for user in users])
    except Exception as e:
        logger.error(f"Error in get_admin_users: {str(e)}")
        return jsonify({'error': 'Failed to fetch users'}), 500

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error="Internal server error"), 500

@app.route('/api/debug/data_status')
@login_required
def debug_data_status():
    """Endpoint to check data availability for debugging purposes"""
    try:
        sales_count = Sale.query.filter_by(user_id=current_user.id).count()
        purchases_count = Purchase.query.filter_by(user_id=current_user.id).count()
        products_count = Product.query.filter_by(user_id=current_user.id).count()
        customers_count = Customer.query.filter_by(user_id=current_user.id).count()
        vendors_count = Vendor.query.filter_by(user_id=current_user.id).count()
        
        return jsonify({
            'status': 'success',
            'data': {
                'sales_count': sales_count,
                'purchases_count': purchases_count,
                'products_count': products_count,
                'customers_count': customers_count,
                'vendors_count': vendors_count,
                'user_id': current_user.id,
                'user_email': current_user.email,
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Error in debug_data_status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
