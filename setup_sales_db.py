#####################
# Run this script once to create a realistic sales database with 200 customers, 500 sales, and 20 products. 
# The database will be saved at 'sales_db/sales_data.db'
#####################

import os
from pathlib import Path
import sqlite3
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
Faker.seed(42)  # Reproducible data
random.seed(42)

BASE_DIR = Path(__file__).parent  
DB_PATH = str(BASE_DIR / "sales_db" / "sales_data.db")

def create_database():
    """Create sales database with realistic schema"""
    
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop existing tables
    cursor.execute("DROP TABLE IF EXISTS sales_items")
    cursor.execute("DROP TABLE IF EXISTS sales")
    cursor.execute("DROP TABLE IF EXISTS customers")
    cursor.execute("DROP TABLE IF EXISTS products")
    cursor.execute("DROP TABLE IF EXISTS regions")
    
    # Create tables
    cursor.execute("""
        CREATE TABLE regions (
            region_id INTEGER PRIMARY KEY,
            region_name TEXT NOT NULL,
            country TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            email TEXT,
            company TEXT,
            region_id INTEGER,
            customer_since DATE,
            customer_tier TEXT,
            FOREIGN KEY (region_id) REFERENCES regions(region_id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            unit_price REAL NOT NULL,
            cost REAL NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE sales (
            sale_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            sale_date DATE NOT NULL,
            total_amount REAL NOT NULL,
            status TEXT NOT NULL,
            sales_rep TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE sales_items (
            item_id INTEGER PRIMARY KEY,
            sale_id INTEGER,
            product_id INTEGER,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            discount REAL DEFAULT 0,
            FOREIGN KEY (sale_id) REFERENCES sales(sale_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )
    """)
    
    conn.commit()
    return conn

def populate_regions(conn):
    """Add regions"""
    regions = [
        (1, "North America", "USA"),
        (2, "Europe", "UK"),
        (3, "Europe", "Germany"),
        (4, "Asia Pacific", "Singapore"),
        (5, "Asia Pacific", "Australia"),
        (6, "South America", "Brazil"),
    ]
    
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO regions VALUES (?, ?, ?)",
        regions
    )
    conn.commit()
    print(f"✓ Created {len(regions)} regions")

def populate_products(conn):
    """Add products"""
    categories = {
        "Software": [
            ("Enterprise Suite Pro", 299.99, 50.00),
            ("Analytics Dashboard", 199.99, 30.00),
            ("Security Plus", 149.99, 25.00),
            ("Cloud Storage Premium", 99.99, 15.00),
        ],
        "Hardware": [
            ("Server Rack Unit", 2499.99, 1200.00),
            ("Network Router Pro", 899.99, 400.00),
            ("Workstation Elite", 1599.99, 800.00),
            ("Storage Drive 2TB", 299.99, 150.00),
        ],
        "Services": [
            ("Consulting Hours", 150.00, 75.00),
            ("Training Session", 500.00, 200.00),
            ("Support Package", 1200.00, 300.00),
            ("Implementation Service", 3000.00, 1500.00),
        ],
        "Licenses": [
            ("Annual License", 599.99, 100.00),
            ("Enterprise License", 2999.99, 500.00),
            ("Developer License", 399.99, 80.00),
            ("Team License (5 users)", 999.99, 200.00),
        ]
    }
    
    products = []
    product_id = 1
    for category, items in categories.items():
        for name, price, cost in items:
            products.append((product_id, name, category, price, cost))
            product_id += 1
    
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?)",
        products
    )
    conn.commit()
    print(f"✓ Created {len(products)} products")
    return products

def populate_customers(conn, num_customers=200):
    """Add customers"""
    tiers = ["Bronze", "Silver", "Gold", "Platinum"]
    
    customers = []
    for i in range(1, num_customers + 1):
        customer_name = fake.name()
        email = fake.email()
        company = fake.company()
        region_id = random.randint(1, 6)
        days_ago = random.randint(30, 1095)  # 1 month to 3 years
        customer_since = (datetime.now() - timedelta(days=days_ago)).date()
        tier = random.choices(tiers, weights=[40, 35, 20, 5])[0]
        
        customers.append((i, customer_name, email, company, region_id, customer_since, tier))
    
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?)",
        customers
    )
    conn.commit()
    print(f"✓ Created {num_customers} customers")
    return customers

def populate_sales(conn, num_sales=500, products=None):
    """Add sales transactions"""
    if products is None:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM products")
        products = cursor.fetchall()
    
    cursor = conn.cursor()
    cursor.execute("SELECT customer_id FROM customers")
    customer_ids = [row[0] for row in cursor.fetchall()]
    
    sales_reps = ["Alice Johnson", "Bob Smith", "Carol Williams", "David Brown", "Eve Davis"]
    statuses = ["Completed", "Pending", "Cancelled"]
    
    sales_data = []
    items_data = []
    
    sale_id = 1
    item_id = 1
    
    # Generate fake sales over past 18 months
    start_date = datetime.now() - timedelta(days=540)
    
    for _ in range(num_sales):
        # Random date in past 18 months
        days_offset = random.randint(0, 540)
        sale_date = (start_date + timedelta(days=days_offset)).date()
        
        customer_id = random.choice(customer_ids)
        status = random.choices(statuses, weights=[85, 10, 5])[0]
        sales_rep = random.choice(sales_reps)
        
        # Each sale has 1-5 items
        num_items = random.choices([1, 2, 3, 4, 5], weights=[40, 30, 20, 7, 3])[0]
        total_amount = 0
        
        sale_items = []
        for _ in range(num_items):
            product = random.choice(products)
            product_id = product[0]
            base_price = product[3]
            
            quantity = random.choices([1, 2, 3, 5, 10], weights=[50, 25, 15, 7, 3])[0]
            
            # Random discount (0-20%)
            discount = random.choices([0, 0.05, 0.10, 0.15, 0.20], weights=[60, 20, 10, 7, 3])[0]
            
            unit_price = base_price
            item_total = round(unit_price * quantity * (1 - discount), 2)
            total_amount += item_total
            
            sale_items.append((item_id, sale_id, product_id, quantity, unit_price, discount))
            item_id += 1

        total_amount = round(total_amount, 2)
        sales_data.append((sale_id, customer_id, sale_date, total_amount, status, sales_rep))
        items_data.extend(sale_items)
        sale_id += 1
    
    cursor.executemany(
        "INSERT INTO sales VALUES (?, ?, ?, ?, ?, ?)",
        sales_data
    )
    
    cursor.executemany(
        "INSERT INTO sales_items VALUES (?, ?, ?, ?, ?, ?)",
        items_data
    )
    
    conn.commit()
    print(f"✓ Created {num_sales} sales with {len(items_data)} line items")

def create_indexes(conn):
    """Create indexes for better query performance"""
    cursor = conn.cursor()
    
    cursor.execute("CREATE INDEX idx_sales_date ON sales(sale_date)")
    cursor.execute("CREATE INDEX idx_sales_customer ON sales(customer_id)")
    cursor.execute("CREATE INDEX idx_sales_status ON sales(status)")
    cursor.execute("CREATE INDEX idx_items_sale ON sales_items(sale_id)")
    cursor.execute("CREATE INDEX idx_items_product ON sales_items(product_id)")
    cursor.execute("CREATE INDEX idx_customers_region ON customers(region_id)")
    
    conn.commit()
    print("✓ Created indexes")

def print_summary(conn):
    """Print database summary"""
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("DATABASE SUMMARY")
    print("="*60)
    
    # Total sales
    cursor.execute("SELECT COUNT(*), SUM(total_amount) FROM sales WHERE status = 'Completed'")
    count, total = cursor.fetchone()
    print(f"Total Completed Sales: {count}")
    print(f"Total Revenue: ${total:,.2f}")
    
    # By region
    cursor.execute("""
        SELECT r.region_name, COUNT(s.sale_id), SUM(s.total_amount)
        FROM sales s
        JOIN customers c ON s.customer_id = c.customer_id
        JOIN regions r ON c.region_id = r.region_id
        WHERE s.status = 'Completed'
        GROUP BY r.region_name
        ORDER BY SUM(s.total_amount) DESC
    """)
    print("\nRevenue by Region:")
    for region, count, total in cursor.fetchall():
        print(f"  {region:20s}: ${total:>12,.2f} ({count} sales)")
    
    # Top products
    cursor.execute("""
        SELECT p.product_name, COUNT(si.item_id), SUM(si.quantity * si.unit_price * (1 - si.discount))
        FROM sales_items si
        JOIN products p ON si.product_id = p.product_id
        JOIN sales s ON si.sale_id = s.sale_id
        WHERE s.status = 'Completed'
        GROUP BY p.product_name
        ORDER BY SUM(si.quantity * si.unit_price * (1 - si.discount)) DESC
        LIMIT 5
    """)
    print("\nTop 5 Products by Revenue:")
    for product, count, total in cursor.fetchall():
        print(f"  {product:30s}: ${total:>12,.2f} ({count} sold)")
    
    print("="*60 + "\n")

def main():
    print("🗄️  Creating Sales Database...")
    print("="*60)
    
    conn = create_database()
    
    populate_regions(conn)
    products = populate_products(conn)
    populate_customers(conn, num_customers=200)
    populate_sales(conn, num_sales=500, products=products)
    create_indexes(conn)
    
    print_summary(conn)
    
    conn.close()
    print(f"✅ Database created successfully: {DB_PATH}")

if __name__ == "__main__":
    main()