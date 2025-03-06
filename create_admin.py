from app import app, db
from models import User
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_admin():
    with app.app_context():
        try:
            # Check if admin user exists
            admin_email = "forecastai007@gmail.com"
            admin = User.query.filter_by(email=admin_email).first()
            
            if admin:
                logger.info(f"Admin user found: {admin.email}, is_admin: {admin.is_admin}")
                
                if not admin.is_admin:
                    admin.is_admin = True
                    db.session.commit()
                    logger.info("Existing user updated to admin status")
            else:
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
                
            # Verify admin was created/updated correctly
            admin = User.query.filter_by(email=admin_email).first()
            if admin and admin.is_admin:
                logger.info("Admin user verification successful")
                print("Admin user created/verified successfully!")
                print(f"Email: {admin_email}")
                print("Password: Forecast@007")
            else:
                logger.error("Admin user verification failed")
                print("Failed to create/verify admin user")
                
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating admin user: {str(e)}")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    create_admin()

