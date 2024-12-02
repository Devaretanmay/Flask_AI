# Energy Analytics Dashboard

A Flask-based web application for visualizing and analyzing energy consumption data using natural language queries.

## Features

- Natural Language Query Interface
- Pre-written Analysis Queries
- Interactive Data Visualization
- Real-time Data Analysis
- Dark/Light Theme Support

## Tech Stack

- Backend: Flask (Python)
- Database: PostgreSQL
- Query Generation: Vanna AI
- Visualization: Plotly.js
- Frontend: HTML/CSS/JavaScript

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd vanna-flask
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
```
VANNA_API_KEY=your_vanna_api_key
VANNA_MODEL=your_model_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=your_db_host
DB_PORT=your_db_port
DB_NAME=your_db_name
```

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Project Structure

```
vanna-flask/
├── app.py              # Main Flask application
├── templates/          # HTML templates
│   └── index.html     # Main dashboard template
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
└── README.md          # Project documentation
```

## Usage

1. Open the application in your browser
2. Use pre-written queries from the sidebar
3. Or type your own questions in natural language
4. View results in both table and graph format

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - feel free to use this project for your own purposes.
