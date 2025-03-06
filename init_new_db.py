from app import app, db
from models import Customer, Product, Sale, SaleItem, Vendor, Purchase, PurchaseItem, User
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    with app.app_context():
        try:
            # Create tables
            db.create_all()
            logger.info("Database tables created successfully")
            
            # Check if we already have data
            if User.query.count() > 0:
                logger.info("Database already initialized")
                return
            
            # Create admin user if it doesn't exist
            admin_email = "forecastai007@gmail.com"
            admin = User.query.filter_by(email=admin_email).first()
            
            if not admin:
                admin = User(
                    name="admin_Varshit_k",
                    email=admin_email,
                    is_admin=True
                )
                admin.set_password("Forecast@007")
                db.session.add(admin)
                db.session.commit()
                logger.info("Admin user created")
            
            # Add sample customers
            customers = [
                Customer(
                    user_id=admin.id,
                    gst_id="29AABCU9603R1ZJ", 
                    name="ABC Enterprises", 
                    contact_person="Rajesh Kumar", 
                    phone="9876543210", 
                    location="Bangalore", 
                    about="Regular customer for electronics and IT equipment. Prefers bulk orders with advance payment."
                ),
                Customer(
                    user_id=admin.id,
                    gst_id="27AADCS0472N1Z1", 
                    name="XYZ Corporation", 
                    contact_person="Sunil Mehta", 
                    phone="8765432109", 
                    location="Mumbai", 
                    about="Corporate client with monthly purchase requirements. Credit period of 30 days."
                ),
                Customer(
                    user_id=admin.id,
                    gst_id="33AACFD4893J1ZK", 
                    name="PQR Limited", 
                    contact_person="Anita Sharma", 
                    phone="7654321098", 
                    location="Chennai", 
                    about="New customer with growing business. Interested in smartphones and tablets."
                )
            ]
            
            db.session.add_all(customers)
            db.session.commit()
            logger.info("Sample customers added")
            
            # Add sample vendors
            vendors = [
                Vendor(
                    user_id=admin.id,
                    gst_id="19AAACP5773D1ZT", 
                    name="Tech Suppliers", 
                    contact_person="Rahul Sharma", 
                    phone="9876543210", 
                    location="Delhi", 
                    about="IT hardware and accessories supplier. Offers 15 days credit period."
                ),
                Vendor(
                    user_id=admin.id,
                    gst_id="36AADCS1234A1Z9", 
                    name="Office Solutions", 
                    contact_person="Priya Patel", 
                    phone="8765432109", 
                    location="Hyderabad", 
                    about="Office furniture and stationery supplier. Provides free delivery for orders above ₹10,000."
                ),
                Vendor(
                    user_id=admin.id,
                    gst_id="24AAACR9876B1Z5", 
                    name="Electronics Hub", 
                    contact_person="Amit Kumar", 
                    phone="7654321098", 
                    location="Pune", 
                    about="Electronic components and devices supplier. Known for quality products and timely delivery."
                )
            ]
            
            db.session.add_all(vendors)
            db.session.commit()
            logger.info("Sample vendors added")
            
            # Add sample products
            products = [
                Product(user_id=admin.id, product_id="PRD001", name="Laptop", quantity=50, cost_per_unit=45000, specifications="15.6 inch, 8GB RAM, 512GB SSD, Intel Core i5"),
                Product(user_id=admin.id, product_id="PRD002", name="Smartphone", quantity=100, cost_per_unit=15000, specifications="6.5 inch display, 6GB RAM, 128GB storage, 48MP camera"),
                Product(user_id=admin.id, product_id="PRD003", name="Tablet", quantity=30, cost_per_unit=20000, specifications="10.1 inch, 4GB RAM, 64GB storage, WiFi + 4G"),
                Product(user_id=admin.id, product_id="PRD004", name="Monitor", quantity=40, cost_per_unit=12000, specifications="24 inch, Full HD, IPS panel, HDMI + VGA ports"),
                Product(user_id=admin.id, product_id="PRD005", name="Keyboard", quantity=80, cost_per_unit=1500, specifications="Mechanical keyboard, RGB backlight, USB interface")
            ]
            
            db.session.add_all(products)
            db.session.commit()
            logger.info("Sample products added")
            
            # Add sample purchases
            today = datetime.now()
            purchases = [
                Purchase(
                    user_id=admin.id,
                    vendor_id=1,
                    order_id="PO-2025-001",
                    purchase_date=today - timedelta(days=30),
                    delivery_charges=500,
                    total_amount=50500,
                    status="Delivered"
                ),
                Purchase(
                    user_id=admin.id,
                    vendor_id=2,
                    order_id="PO-2025-002",
                    purchase_date=today - timedelta(days=15),
                    delivery_charges=750,
                    total_amount=75750,
                    status="In Transit"
                ),
                Purchase(
                    user_id=admin.id,
                    vendor_id=3,
                    order_id="PO-2025-003",
                    purchase_date=today - timedelta(days=60),
                    delivery_charges=350,
                    total_amount=35350,
                    status="Delivered"
                )
            ]
            
            db.session.add_all(purchases)
            db.session.commit()
            logger.info("Sample purchases added")
            
            # Add sample purchase items
            purchase_items = [
                PurchaseItem(
                    purchase_id=1,
                    product_id=1,
                    quantity=1,
                    gst_percentage=18,
                    unit_price=42500,
                    total_price=50150
                ),
                PurchaseItem(
                    purchase_id=2,
                    product_id=2,
                    quantity=5,
                    gst_percentage=18,
                    unit_price=12500,
                    total_price=73750
                ),
                PurchaseItem(
                    purchase_id=3,
                    product_id=5,
                    quantity=20,
                    gst_percentage=18,
                    unit_price=1500,
                    total_price=35400
                )
            ]
            
            db.session.add_all(purchase_items)
            db.session.commit()
            logger.info("Sample purchase items added")
            
            logger.info("Database initialized with sample data")
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error initializing database: {str(e)}")
            raise

if __name__ == "__main__":
    init_db()

