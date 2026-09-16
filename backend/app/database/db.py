import mysql.connector
from mysql.connector import Error

try:
    # 1. Establish the connection to the MySQL server
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="your_password",      # Replace with your actual password
        database="my_project_db"       # Ensure this database exists
    )


except Error as e:
    print(f"Error while connecting to MySQL: {e}")

finally:
    # 5. Clean up and close connections safely
    if 'connection' in locals() and connection.is_connected():
        connection.close()
        print("MySQL connection closed.")
