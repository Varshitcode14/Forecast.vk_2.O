from app import app, db
from models import Customer, Product, Sale, SaleItem, Vendor, Purchase, PurchaseItem
from datetime import datetime, timedelta

def init_db():
    with app.app_context():
        # Create tables
        db.create_all()
        
        # Check if we already have data
        if Customer.query.count() > 0:
            print("Database already initialized")
            return
        
        # Add sample customers
        customers = [
            Customer(
                gst_id="29AABCU9603R1ZJ", 
                name="ABC Enterprises", 
                contact_person="Rajesh Kumar", 
                phone="9876543210", 
                location="Bangalore", 
                about="Regular customer for electronics and IT equipment. Prefers bulk orders with advance payment."
            ),
            Customer(
                gst_id="27AADCS0472N1Z1", 
                name="XYZ Corporation", 
                contact_person="Sunil Mehta", 
                phone="8765432109", 
                location="Mumbai", 
                about="Corporate client with monthly purchase requirements. Credit period of 30 days."
            ),
            Customer(
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
        
        # Add sample vendors
        vendors = [
            Vendor(
                gst_id="19AAACP5773D1ZT", 
                name="Tech Suppliers", 
                contact_person="Rahul Sharma", 
                phone="9876543210", 
                location="Delhi", 
                about="IT hardware and accessories supplier. Offers 15 days credit period."
            ),
            Vendor(
                gst_id="36AADCS1234A1Z9", 
                name="Office Solutions", 
                contact_person="Priya Patel", 
                phone="8765432109", 
                location="Hyderabad", 
                about="Office furniture and stationery supplier. Provides free delivery for orders above ₹10,000."
            ),
            Vendor(
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
        
        # Add sample products
        products = [
            Product(product_id="PRD001", name="Laptop", quantity=50, cost_per_unit=45000, specifications="15.6 inch, 8GB RAM, 512GB SSD, Intel Core i5"),
            Product(product_id="PRD002", name="Smartphone", quantity=100, cost_per_unit=15000, specifications="6.5 inch display, 6GB RAM, 128GB storage, 48MP camera"),
            Product(product_id="PRD003", name="Tablet", quantity=30, cost_per_unit=20000, specifications="10.1 inch, 4GB RAM, 64GB storage, WiFi + 4G"),
            Product(product_id="PRD004", name="Monitor", quantity=40, cost_per_unit=12000, specifications="24 inch, Full HD, IPS panel, HDMI + VGA ports"),
            Product(product_id="PRD005", name="Keyboard", quantity=80, cost_per_unit=1500, specifications="Mechanical keyboard, RGB backlight, USB interface")
        ]
        
        db.session.add_all(products)
        db.session.commit()
        
        # Add sample purchases
        today = datetime.now()
        purchases = [
            Purchase(
                vendor_id=1,
                order_id="PO-2025-001",
                purchase_date=today - timedelta(days=30),
                delivery_charges=500,
                total_amount=50500,
                status="Delivered"
            ),
            Purchase(
                vendor_id=2,
                order_id="PO-2025-002",
                purchase_date=today - timedelta(days=15),
                delivery_charges=750,
                total_amount=75750,
                status="In Transit"
            ),
            Purchase(
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
        
        print("Database initialized with sample data")

if __name__ == "__main__":
    init_db()

