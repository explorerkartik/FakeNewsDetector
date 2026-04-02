import psycopg2

conn = psycopg2.connect('postgres://avnadmin:AVNS_Cq5FyI0livtu_ypdkGZ@pg-9c5f9fb-tiwarikartik43-3387.e.aivencloud.com:18975/defaultdb?sslmode=require')
cur = conn.cursor()

cur.execute('''CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(150) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)''')

cur.execute('''CREATE TABLE IF NOT EXISTS analysis_history (
    id SERIAL PRIMARY KEY,
    user_id INT,
    input_text TEXT,
    verdict VARCHAR(10),
    credibility_score INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
)''')

conn.commit()
conn.close()
print('Tables created successfully!')