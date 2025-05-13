# backend/app.py
import patch  # Import the patching module first

import threading
print("Threading module patched app.py:", "GreenThread" in dir(threading))

import pickle
import signal
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from dotenv import load_dotenv
from datetime import datetime
import os
import sqlite3
import MetaTrader5 as mt5
import logging
from logging.handlers import RotatingFileHandler

from database import init_db
from deploy import (
    get_mt5_info,
    initialize_mt5, 
    RealTimeTrader, 
    get_mt5_account_info,
    decrypt_password
)

load_dotenv()  # Load .env file
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

# Enable CORS for all routes, allowing requests from http://localhost:3000
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# Get environment variables
DATASET_FOLDER = os.getenv("DATASET_FOLDER")
CONFIG_FOLDER = os.getenv("CONFIG_FOLDER")

# Global variables
is_trading_running = False
stop_event = threading.Event()
trader_instance = None  # To hold the RealTimeTrader instance

# Configure logging with a file handler
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api.log')
handler = RotatingFileHandler(
    log_file,
    maxBytes=1024 * 1024,  # 1 MB
    backupCount=5  # Keep 5 backup files
)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Get the root logger and clear any existing handlers
logger = logging.getLogger()
logger.handlers = []  # Clear default handlers
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Emit status update
def emit_status():
    socketio.emit("status_update", {"isRunning": is_trading_running})

def get_db_connection():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database.db")
    conn = sqlite3.connect(db_path)
    return conn

def run_trading(trader):
    """Wrapper function to run the trading loop with stop control"""
    global is_trading_running
    try:
        logging.info("Trading thread started")
        trader.run()  # Call the run method of RealTimeTrader
        logging.info("Trading thread stopped normally")
    except Exception as e:
        logging.error(f"Trading loop error: {str(e)}")
    finally:
        is_trading_running = False
        emit_status()  # Notify clients when thread stops

@app.route('/api/mt5-login', methods=['POST'])
def mt5_login():
    try:
        # Get JSON data from the frontend
        data = request.get_json()
        if not data:
            logging.error("No data provided in mt5-login request")
            return jsonify({"error": "No data provided"}), 400

        account = data.get("account")
        password = data.get("password")
        server = data.get("server")

        # Validate input
        if not all([account, password, server]):
            logging.error("Missing required parameters in mt5-login")
            return jsonify({"error": "Missing required parameters: account, password, or server"}), 400

        # Convert account to integer (MetaTrader expects an int)
        try:
            account = int(account)
        except ValueError:
            logging.error(f"Invalid account value: {account}")
            return jsonify({"error": "Account must be a valid integer"}), 400

        # Decrypt the password
        try:
            plaintext_password = decrypt_password(password)
        except ValueError as e:
            logging.error(f"Password decryption failed: {str(e)}")
            return jsonify({"error": str(e)}), 400
        
        # print(f"Account: {account}, Password: {plaintext_password}, Server: {server}")
        # Call the MT5 initialization function
        success = initialize_mt5(account, plaintext_password, server)
        if success:
            logging.info(f"Successfully logged into MT5 with account {account}")
            return jsonify({"message": f"Successfully logged into MT5 with account {account}"}), 200
        else:
            logging.error(f"Failed to initialize MT5 for account {account}")
            return jsonify({"error": "Failed to initialize or login to MT5"}), 500

    except Exception as e:
        logging.error(f"Server error in mt5-login: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/mt5-account-info', methods=['GET'])
def mt5_account_info():
    try:
        # Get userId, account, server from query parameters
        user_id = request.args.get('userId')
        account = request.args.get('account')
        encrypted_password = request.args.get('password')
        server = request.args.get('server')

        if not user_id or not account or not server:
            logging.error("Missing query parameters in mt5-account-info")
            return jsonify({"error": "userId or account or server query parameter is required"}), 400
        
        # Decrypt the password
        try:
            plaintext_password = decrypt_password(encrypted_password)
        except ValueError as e:
            logging.error(f"Password decryption failed: {str(e)}")
            return jsonify({"error": str(e)}), 400

        # Get account info after reinitialization
        account_info = get_mt5_account_info(int(account), plaintext_password, server)
        if account_info:
            logging.info(f"Retrieved account info for account {account}")
            return jsonify({
                "message": "Successfully retrieved MT5 account information",
                "account_info": account_info
            }), 200
        else:
            logging.error(f"Failed to retrieve account info for account {account}")
            return jsonify({
                "error": "Failed to retrieve account information. MT5 may not be initialized or logged in."
            }), 400

    except Exception as e:
        logging.error(f"Error in mt5_account_info: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/trading/start', methods=['POST'])
def start_trading():
    global is_trading_running, trader_instance
    
    # Get JSON data from the frontend
    data = request.get_json()
    if not data:
        logging.error("No JSON data received in start_trading")
        return jsonify({"error": "Request must contain JSON data"}), 400

    accountId = data.get("accountId")
    account = data.get("account")
    password = data.get("password")
    server = data.get("server")
    model = data.get("model") 
    
    # Validate input
    if not all([accountId, account, password, server]):
        logging.error("Missing required parameters in start_trading")
        return jsonify({"error": "Missing required parameters: accountId, account, password, or server"}), 400

    # Convert account to integer (MetaTrader expects an int)
    try:
        account = int(account)
    except ValueError:
        logging.error(f"Invalid account value: {account}")
        return jsonify({"error": "Account must be a valid integer"}), 400

    # Decrypt the password
    try:
        plaintext_password = decrypt_password(password)
    except ValueError as e:
        logging.error(f"Password decryption failed: {str(e)}")
        return jsonify({"error": str(e)}), 400

    if is_trading_running:
        logging.info("Trading already running, rejecting request")
        return jsonify({
            'status': 'error',
            'message': 'Trading is already running'
        }), 400
    
    try:
        logging.info(f"Starting trading with model: {model}, accountId: {accountId}")

        # Create a new RealTimeTrader instance
        trader_instance = RealTimeTrader(
            account_id=accountId,  # Convert to string if needed by RealTimeTrader
            login_id=int(account),   
            password=plaintext_password,
            server=server,
            symbol="XAUUSD",          # Default symbol, could be made configurable
            stop_event=stop_event     # Pass the global stop_event
        )

        # Clear the stop event and start new thread
        stop_event.clear()
        trader_instance.start()  # Start both trading and monitoring threads
        is_trading_running = True
        emit_status()  # Notify clients
        
        logging.info(f"Trading started at {datetime.now()}")
        return jsonify({
            'status': 'success',
            'message': 'Trading and trade monitoring started successfully',
            'model': model,
            'accountId': accountId
        }), 200
    except Exception as e:
        logging.error(f"Error starting trading: {str(e)}")
        is_trading_running = False
        return jsonify({
            'status': 'error',
            'message': f"Failed to start trading: {str(e)}"
        }), 500

@app.route('/api/trading/stop', methods=['POST'])
def stop_trading():
    global is_trading_running, trader_instance
    
    if not is_trading_running:
        logging.info("Trading not running, nothing to stop")
        return jsonify({'status': 'error', 'message': 'Trading is not running'}), 400
    
    if trader_instance:
        trader_instance.stop()  # Stop both threads via RealTimeTrader
    
    is_trading_running = False
    logging.info(f"Trading stopped at {datetime.now()}")
    emit_status()  # Notify clients
    return jsonify({'status': 'success', 'message': 'Trading and monitoring stopped successfully'}), 200

@app.route('/api/trading/status', methods=['GET'])
def get_trading_status():
    logging.info("Checking trading status")
    return jsonify({
        'status': 'success',
        'isRunning': is_trading_running,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/mt5/info', methods=['GET'])
def mt5_info():
    logging.info("Get MT5 info")
    
    # Get query parameters
    account = request.args.get('account')
    encrypted_password = request.args.get('password')
    server = request.args.get('server')
    symbol = request.args.get('symbol', 'XAUUSD')  # Default to XAUUSD if not provided

    # Validate query parameters
    if not account or not server or not encrypted_password:
        logging.error("Missing query parameters in mt5-info")
        return jsonify({"error": "account, server, and password query parameters are required"}), 400
    
    # Convert account to integer
    try:
        account = int(account)
    except ValueError:
        logging.error("Account must be an integer")
        return jsonify({"error": "account must be an integer"}), 400

    # Decrypt the password
    try:
        plaintext_password = decrypt_password(encrypted_password)
    except ValueError as e:
        logging.error(f"Password decryption failed: {str(e)}")
        return jsonify({"error": f"Password decryption failed: {str(e)}"}), 400

    # Initialize MT5
    if not initialize_mt5(account, plaintext_password, server):
        logging.error(f"Failed to initialize MT5 for account {account}")
        return jsonify({
            "error": "Failed to initialize MT5 or login. Check credentials or server."
        }), 500

    try:
        # Fetch MT5 data
        data = get_mt5_info(symbol)
        
        # Check if data contains an error
        if isinstance(data, dict) and "error" in data:
            return jsonify({
                "error": data["error"]
            }), 400

        logging.info(f"Successfully retrieved MT5 info for symbol {symbol}")
        return jsonify({
            "status": "success",
            "data": data,
        }), 200

    except Exception as e:
        logging.error(f"Error in mt5_info: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/event-map')
def get_event_map():
    try:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/event_map.pkl')
        if not os.path.exists(file_path):
            logging.error(f"Event map file not found: {file_path}")
            return jsonify({"error": "Event map file not found"}), 404
        with open(file_path, 'rb') as f:
            event_map = pickle.load(f)
        logging.info("Successfully loaded event map")
        return jsonify(event_map), 200
    except Exception as e:
        logging.error(f"Error loading event map: {str(e)}")
        return jsonify({"error": f"Failed to load event map: {str(e)}"}), 500

@app.route('/api/currency-map')
def get_currency_map():
    try:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/currency_map.pkl')
        if not os.path.exists(file_path):
            logging.error(f"Currency map file not found: {file_path}")
            return jsonify({"error": "Currency map file not found"}), 404
        with open(file_path, 'rb') as f:
            currency_map = pickle.load(f)
        logging.info("Successfully loaded currency map")
        return jsonify(currency_map), 200
    except Exception as e:
        logging.error(f"Error loading currency map: {str(e)}")
        return jsonify({"error": f"Failed to load currency map: {str(e)}"}), 500

@app.route('/api/algo-trading-status', methods=['POST'])
def check_algo_trading_status():
    try:
        data = request.get_json()
        account = data.get('account')
        password = data.get('password')
        server = data.get('server')

        if not all([account, password, server]):
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        # Initialize MT5 connection
        if not initialize_mt5(account=int(account), password=decrypt_password(password), server=server):
            return jsonify({"status": "error", "message": "Failed to initialize MT5 connection"}), 500

        # Check algo trading status
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            mt5.shutdown()
            return jsonify({"status": "error", "message": "Failed to get terminal info"}), 500

        is_algo_enabled = terminal_info.trade_allowed

        return jsonify({
            "status": "success",
            "isAlgoTradingEnabled": is_algo_enabled
        })
    except Exception as e:
        logging.error(f"Error checking algo trading status: {str(e)}")
        mt5.shutdown()
        return jsonify({"status": "error", "message": f"Error checking algo trading status: {str(e)}"}), 500
        
# Error handlers
@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"API error: {str(e)}")
    return jsonify({
        'status': 'error',
        'message': 'An unexpected error occurred'
    }), 500

# Cleanup function to run on shutdown
def cleanup():
    global is_trading_running, trader_instance
    logging.info("Cleaning up before shutdown...")

    # Stop trading if running
    if is_trading_running and trader_instance:
        logging.info("Stopping RealTimeTrader...")
        trader_instance.stop()
        is_trading_running = False

    # Stop Flask-SocketIO and ensure eventlet green threads are cleaned up
    logging.info("Stopping Flask-SocketIO...")
    try:
        socketio.stop()
        # Use patch.eventlet_module.sleep() instead of eventlet.sleep()
        patch.eventlet_module.sleep(1)  # Give eventlet a moment to finish any remaining green threads
        logging.info("Flask-SocketIO stopped successfully")
    except Exception as e:
        logging.error(f"Error stopping Flask-SocketIO: {str(e)}")

    # Close logging handlers
    logging.info("Closing logging handlers...")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    logging.info("Cleanup completed, exiting...")
    # Use os._exit(0) for a more forceful shutdown to ensure the process terminates
    os._exit(0)

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    logging.info(f"Received signal {sig}, shutting down...")
    cleanup()

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    # Initialize SQLite database (assumes init_db is safe to call multiple times)
    init_db()
    logging.info("Database initialized")
    try:
        # Change the working directory to avoid locking the backend folder
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        logging.info(f"Changed working directory to: {os.getcwd()}")
        socketio.run(app, host="127.0.0.1", port=int(os.getenv("BACKEND_PORT", 5000)), use_reloader=False, debug=False)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, shutting down...")
        cleanup()
    except Exception as e:
        logging.error(f"Server error: {str(e)}")
        cleanup()