from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class CoffeeProduct(Base):
    __tablename__ = 'coffee_products'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    
    # Связь
    sales = relationship("CoffeeSale", back_populates="product")
    
    def __repr__(self):
        return f"<CoffeeProduct(id={self.id}, name='{self.name}')>"

class CoffeeSale(Base):
    __tablename__ = 'coffee_sales'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('coffee_products.id'))
    sale_date = Column(DateTime, nullable=False)
    timestamp = Column(DateTime, nullable=True)  # Полная временная метка, если доступна
    payment_type = Column(String(50))
    price = Column(Float, nullable=False)  # Цена кофе
    
    # Связь
    product = relationship("CoffeeProduct", back_populates="sales")
    
    def __repr__(self):
        return f"<CoffeeSale(id={self.id}, date='{self.sale_date}', price={self.price})>"

class Customer(Base):
    __tablename__ = 'customers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True)
    address = Column(String(255))
    phone = Column(String(20))
    
    # Связь
    sales = relationship("Sale", back_populates="customer")
    
    def __repr__(self):
        return f"<Customer(id={self.id}, name='{self.name}')>"

class Product(Base):
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(String(255))
    price = Column(Float, nullable=False)
    category = Column(String(50))
    brand = Column(String(50))
    sku = Column(String(50), unique=True)
    
    # Связь
    sales = relationship("Sale", back_populates="product")
    
    def __repr__(self):
        return f"<Product(id={self.id}, name='{self.name}', price={self.price})>"

class Sale(Base):
    __tablename__ = 'sales'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id'))
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=True)
    sale_date = Column(DateTime, nullable=False)
    quantity = Column(Integer, default=1)
    price = Column(Float, nullable=False)  # Цена на момент продажи
    total_amount = Column(Float, nullable=False)
    
    # Связи
    product = relationship("Product", back_populates="sales")
    customer = relationship("Customer", back_populates="sales")
    items = relationship("SaleItem", back_populates="sale")
    
    def __repr__(self):
        return f"<Sale(id={self.id}, date='{self.sale_date}', amount={self.total_amount})>"

class SaleItem(Base):
    __tablename__ = 'sale_items'
    
    id = Column(Integer, primary_key=True)
    sale_id = Column(Integer, ForeignKey('sales.id'))
    product_id = Column(Integer, ForeignKey('products.id'))
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    
    # Связи
    sale = relationship("Sale", back_populates="items")
    product = relationship("Product")
    
    def __repr__(self):
        return f"<SaleItem(sale_id={self.sale_id}, product_id={self.product_id}, quantity={self.quantity})>" 