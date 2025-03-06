from flask import Flask
from models import db, User, Customer, Product, Sale, SaleItem, Vendor, Purchase, PurchaseItem, Report
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Determine if we're running locally or on Render
is_production = os.environ.get('RENDER', False)

if is_production:
    # Use PostgreSQL database for Render deployment
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'production-key-for-deployment')
    
    # Get database URL from environment variable or use the hardcoded one as fallback
    database_url = os.environ.get('DATABASE_URL', 
        "postgresql://forecast_vk_2_0_user:ShNUFtmifpJMpDeBwQ5RA5lg90qcNonG@dpg-cv4i8aogph6c7390jsug-a.oregon-postgres.render.com/forecast_vk_2_0")
    
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    # Use SQLite for local development
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///forecast.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db.init_app(app)

def check_column_exists(table_name, column_name):
    """Check if a column exists in a table"""
    with app.app_context():
        from sqlalchemy import text
        
        if is_production:
            # PostgreSQL query
            query = text(f"""
                SELECT EXISTS (
                    SELECT 1 
                    FROM information_schema.columns 
                    WHERE table_name='{table_name}' AND column_name='{column_name}'
                );
            """)
        else:
            # SQLite query
            query = text(f"""
                SELECT COUNT(*) 
                FROM pragma_table_info('{table_name}') 
                WHERE name='{column_name}';
            """)
        
        result = db.session.execute(query).scalar()
        return bool(result)

def add_column(table_name, column_name, column_type):
    """Add a column to a table if it doesn't exist"""
    with app.app_context():
        from sqlalchemy import text
        
        if check_column_exists(table_name, column_name):
            logger.info(f"Column {column_name} already exists in table {table_name}")
            return False
        
        # Add the column
        if is_production:
            # PostgreSQL syntax
            query = text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type};")
        else:
            # SQLite syntax
            query = text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type};")
        
        db.session.execute(query)
        db.session.commit()
        logger.info(f"Added column {column_name} to table {table_name}")
        return True

def migrate_database():
    """Run all migrations"""
    with app.app_context():
        try:
            # Create tables if they don't exist
            db.create_all()
            logger.info("Database tables created (if they didn't exist)")
            
            # Add missing columns to Product table
            add_column('product', 'user_id', 'INTEGER')
            
            # Add foreign key constraint if in PostgreSQL
            if is_production:
                try:
                    from sqlalchemy import text
                    # Check if constraint exists first
                    check_constraint = text("""
                        SELECT COUNT(*) FROM information_schema.table_constraints
                        WHERE constraint_name = 'product_user_id_fkey'
                        AND table_name = 'product';
                    """)
                    constraint_exists = db.session.execute(check_constraint).scalar()
                    
                    if not constraint_exists:
                        # Add foreign key constraint
                        add_constraint = text("""
                            ALTER TABLE product 
                            ADD CONSTRAINT product_user_id_fkey 
                            FOREIGN KEY (user_id) REFERENCES "user" (id);
                        """)
                        db.session.execute(add_constraint)
                        db.session.commit()
                        logger.info("Added foreign key constraint for product.user_id")
                except Exception as e:
                    logger.error(f"Error adding foreign key constraint: {str(e)}")
                    db.session.rollback()
            
            # Set default user_id for existing products if needed
            from sqlalchemy import text
            if is_production:
                # Check if there are products with NULL user_id
                null_check = text("SELECT COUNT(*) FROM product WHERE user_id IS NULL;")
                null_count = db.session.execute(null_check).scalar()
                
                if null_count > 0:
                    # Get the admin user
                    admin = User.query.filter_by(email="forecastai007@gmail.com").first()
                    if admin:
                        # Update all products with NULL user_id to use admin's ID
                        update_query = text(f"UPDATE product SET user_id = {admin.id} WHERE user_id IS NULL;")
                        db.session.execute(update_query)
                        db.session.commit()
                        logger.info(f"Updated {null_count} products with admin user_id")
            
            logger.info("Database migration completed successfully")
            
        except Exception as e:
            logger.error(f"Error during migration: {str(e)}")
            db.session.rollback()
            raise

if __name__ == "__main__":
    migrate_database()

