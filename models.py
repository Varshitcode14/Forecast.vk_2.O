from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)  # Add this line
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'is_admin': self.is_admin
        }

class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    gst_id = db.Column(db.String(15), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    contact_person = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    location = db.Column(db.String(200), nullable=False)
    about = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    sales = db.relationship('Sale', backref='customer', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'gst_id': self.gst_id,
            'name': self.name,
            'contact_person': self.contact_person,
            'phone': self.phone,
            'location': self.location,
            'about': self.about
        }

class Vendor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    gst_id = db.Column(db.String(15), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    contact_person = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    location = db.Column(db.String(200), nullable=False)
    about = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    purchases = db.relationship('Purchase', backref='vendor', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'gst_id': self.gst_id,
            'name': self.name,
            'contact_person': self.contact_person,
            'phone': self.phone,
            'location': self.location,
            'about': self.about
        }

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    quantity = db.Column(db.Integer, default=0)
    cost_per_unit = db.Column(db.Float, nullable=False)
    specifications = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    sale_items = db.relationship('SaleItem', backref='product', lazy=True)
    purchase_items = db.relationship('PurchaseItem', backref='product', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'product_id': self.product_id,
            'name': self.name,
            'quantity': self.quantity,
            'cost_per_unit': self.cost_per_unit,
            'specifications': self.specifications
        }

class Sale(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customer.id'), nullable=False)
    sale_date = db.Column(db.DateTime, default=datetime.utcnow)
    delivery_charges = db.Column(db.Float, default=0.0)
    total_amount = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    items = db.relationship('SaleItem', backref='sale', lazy=True, cascade="all, delete-orphan")

class SaleItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sale_id = db.Column(db.Integer, db.ForeignKey('sale.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    gst_percentage = db.Column(db.Float, default=18.0)  # Default GST rate of 18%
    discount_percentage = db.Column(db.Float, default=0.0)
    unit_price = db.Column(db.Float, nullable=False)
    total_price = db.Column(db.Float, nullable=False)  # Price after GST and discount

class Purchase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    vendor_id = db.Column(db.Integer, db.ForeignKey('vendor.id'), nullable=False)
    purchase_date = db.Column(db.DateTime, default=datetime.utcnow)
    order_id = db.Column(db.String(20), nullable=False)
    delivery_charges = db.Column(db.Float, default=0.0)
    total_amount = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), default='Ordered')  # Ordered, In Transit, Delivered, Cancelled
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    items = db.relationship('PurchaseItem', backref='purchase', lazy=True, cascade="all, delete-orphan")

class PurchaseItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    purchase_id = db.Column(db.Integer, db.ForeignKey('purchase.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    gst_percentage = db.Column(db.Float, default=18.0)  # Default GST rate of 18%
    unit_price = db.Column(db.Float, nullable=False)
    total_price = db.Column(db.Float, nullable=False)  # Price after GST

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    report_type = db.Column(db.String(20), nullable=False)  # daily, weekly, monthly, custom
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    total_sales = db.Column(db.Float, nullable=False)
    total_purchases = db.Column(db.Float, nullable=False)
    net_profit = db.Column(db.Float, nullable=False)
    profit_margin = db.Column(db.Float, nullable=False)
    low_stock_count = db.Column(db.Integer, nullable=False)
    out_of_stock_count = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'report_type': self.report_type,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'total_sales': self.total_sales,
            'total_purchases': self.total_purchases,
            'net_profit': self.net_profit,
            'profit_margin': self.profit_margin,
            'low_stock_count': self.low_stock_count,
            'out_of_stock_count': self.out_of_stock_count,
            'created_at': self.created_at.isoformat()
        }

