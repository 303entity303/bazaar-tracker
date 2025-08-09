import os
import json
# CONFIG_FILE = "./config.json"
# def load_api_key():
#     # If config.json exists, load the key
#     if os.path.exists(CONFIG_FILE):
#         with open(CONFIG_FILE, "r") as f:
#             try:
#                 data = json.load(f)
#                 return data.get("api_key")
#             except json.JSONDecodeError as e:
#                 print(f"config.json is invalid or corrupted: {e}")  # Corrupted or empty file
#
#     # If it doesn't exist or isn't valid, ask the user
#     api_key = input("Insert your API KEY: ").strip()
#     with open(CONFIG_FILE, "w") as f:
#         json.dump({"api_key": api_key}, f)
#         print("To start predicting, restart this program.")
#         exit()
#     return api_key
# api_key = load_api_key()

import warnings
import argparse
import sys
import subprocess
import psutil
import signal
import platform

# ==================================================================== CONFIGURATION PARAMETERS ====================================================================
parser = argparse.ArgumentParser(description="Predict the future buy or sell price of an item on Hypixel Bazaar.")
parser.add_argument("mode", nargs="?", choices=["buy", "sell"], help="Select the type of prediction: 'buy' for buy price, 'sell' for sell price.")
parser.add_argument("item_name", nargs="?", help="The name of the item you want to predict the price for (e.g., ENCHANTED_DIAMOND).")
parser.add_argument("time", nargs="?", type=int, help="How far into the future you want the prediction to go (numeric value only, without unit).")
parser.add_argument("time_unit", nargs="?", choices=["s", "m", "h", "d"], help="Time unit for the prediction: 's' = seconds, 'm' for minutes, 'h' = hours, 'd' = days.")
parser.add_argument("--debug", action="store_true", help="Show detailed information about prediction accuracy and internals.")
parser.add_argument('--print-all', action='store_true', help='Print all predicted values instead of only the best one.')
parser.add_argument('--stop_fetcher', action='store_true', help='Stops the fetcher script and exits.')

args = parser.parse_args()

# ========================================================================== IMPORTS ===========================================================================
from statsmodels.tools.sm_exceptions import ValueWarning
false = False
true = True
import requests
import time
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import pmdarima.arima
import math
from decimal import Decimal, getcontext

# ========================================================================== SUPPRESS WARNINGS ==========================================================================
warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings(
    "ignore",
    message=r".*force_all_finite.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)
warnings.filterwarnings(
    "ignore",
    message=r".*Likelihood.*"
)

# ========================================================================== FETCHER HANDLING ==========================================================================
PID_FILE = "fetcher.pid"
if os.path.exists(PID_FILE):
    fetcher_exist = True
else:
    fetcher_exist = False

if args.stop_fetcher:
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            proc = psutil.Process(pid)
            proc.terminate()  # Use proc.kill() if you want to force it
            print(f"✅ Process fetcher.py (PID {pid}) terminated.")
        except FileNotFoundError as e:
            print(f"ℹ️ fetcher.pid not found: {e}")
            # No exit(), it's already handled below
        except ValueError as e:
            print(f"❌ Invalid PID in fetcher.pid: {e}")
            exit()
        except psutil.NoSuchProcess as e:
            print(f"⚠️ No such process with PID in fetcher.pid: {e}")
            # No exit(), but we can continue
        except psutil.AccessDenied as e:
            print(f"❌ Access denied when trying to terminate fetcher process: {e}")
            exit()
        finally:
            os.remove(PID_FILE)
    else:
        print("ℹ️ No file fetcher.pid found, nothing to kill.")
    exit()

FETCH_SCRIPT = "fetcher.py"
# File to save the PID of the fetcher.py process

# Platform-specific arguments
if not fetcher_exist:
    kwargs = {}
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    # Start fetcher.py
    proc = subprocess.Popen([sys.executable, FETCH_SCRIPT], **kwargs)

    # Save the PID to file
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))
    print(f"fetcher.py started")

# ========================================================================= WHICH TO PREDICT =========================================================================
if args.mode == "buy":
    buy_or_sell = "buyPrice"
elif args.mode == "sell":
    buy_or_sell = "sellPrice"
else:
    # If mode is not provided via command line, prompt the user
    while True:
        mode_input = input("Select prediction type (buy/sell): ").strip().lower()
        if mode_input in ["buy", "sell"]:
            buy_or_sell = f"{mode_input}Price"
            break
        else:
            print("Invalid input. Please enter 'buy' or 'sell'.")

# ============================================================================ ITEM ============================================================================
folder = "prices/" # Translated variable name
item_name = args.item_name # Translated variable name

# Prompt for item name if not provided
if not item_name:
    item_name = input("Enter the item name (e.g., ENCHANTED_DIAMOND): ").strip()
    if not item_name:
        print("ERROR: Item name cannot be empty.")
        exit()

try:
    if not os.path.exists(folder):
        os.makedirs(folder)
except PermissionError as e:
    print(f"❌ Permission denied: cannot create directory '{folder}'. {e}")
    exit()
except OSError as e:
    print(f"❌ Failed to create directory '{folder}': {e}")
    exit()

found = False # Translated variable name
for filename in os.listdir(folder): # Translated variable name
    if filename == f"{item_name}.jsonl":
        found = True # Translated variable name
        item_file = f"{item_name}.jsonl" # Translated variable name

if not found:
    print("ERROR: Item not found. Choose an existing one OR use 'python3 predictor.py -h' for help.")
    exit()

# ============================================================================= TIME =============================================================================
time_raw = args.time # Translated variable name
time_unit = args.time_unit # Translated variable name

# Prompt for time and unit if not provided
if time_raw is None or time_unit is None:
    while True:
        try:
            time_input = input("Enter the time into the future for prediction (e.g., 30m for 30 minutes): ").strip()
            if not time_input:
                raise ValueError("Input cannot be empty.")
            # Simple parsing assuming number followed by unit character
            if time_input[-1].lower() in ['s', 'm', 'h', 'd']:
                time_raw = int(time_input[:-1])
                time_unit = time_input[-1].lower()
                if time_unit in ['s', 'm', 'h', 'd'] and time_raw > 0:
                    break
                else:
                    raise ValueError("Invalid time unit or value.")
            else:
                raise ValueError("Time must end with a valid unit (s, m, h, d).")
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")


if time_unit == "s":
    seconds = time_raw
elif time_unit == "m":
    seconds = time_raw * 60
elif time_unit == "h":
    seconds = time_raw * 3600
elif time_unit == "d":
    seconds = time_raw * 86400
else:
    raise ValueError("Time unit not valid. Use 's', 'm', 'h', or 'd'.")

steps = round(seconds / 20)
# print(f"{time_raw}{time_unit} = {seconds} seconds")
# print(f"Result (seconds / 20): {steps}")

try:
    # print("Continue with the prediction...")
    ## Configuration
    getcontext().prec = 50
    product_id = item_name
    interval_seconds = 20
    N_FORECASTS = steps # Translated variable name
    num_values = 0 # Translated variable name
    input_file = f"prices/{product_id}.jsonl"

    def load_data(input_file): # Translated function name
        global num_values # Use translated variable name
        prices, dates = [], []
        try:
            with open(input_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        price = data.get(buy_or_sell) # Use the correct price field
                        timestamp_ms = data.get("timestamp_ms")
                        if price is not None and timestamp_ms is not None:
                            prices.append(float(price))
                            dates.append(datetime.fromtimestamp(timestamp_ms / 1000))
                            num_values += 1 # Increment translated variable
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Skipping invalid JSON line: {e}")
            # No exit(), we can continue
        except FileNotFoundError as e:
            print(f"❌ Price file '{input_file}' not found: {e}")
            exit()
        except PermissionError as e:
            print(f"❌ Cannot read file '{input_file}': {e}")
            exit()
        except OSError as e:
            print(f"❌ Error reading file '{input_file}': {e}")
            exit()
        return pd.Series(prices, index=pd.to_datetime(dates)).sort_index()

    def percent_error(pred, real): # Translated function name
        if real == 0:
            return None
        return round(abs(pred - real) / real * 100, 5)

    def mape(y_true, y_pred): # Mean Absolute Percentage Error
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # Load historical data
    ts = load_data(input_file) # Use translated function name
    if len(ts) < 10:
        print("\033[91mERROR\033[0m Not enough data. Recommended at least 40 values (about 13 minutes worth of data collection).")
        exit()
    elif len(ts) < 40: # Changed to elif for better logic
        print("\033[93mWARNING\033[0m I personally recommend at least 40 values (about 13 minutes worth of data collection to predict about 30 minutes).")

    # Perform prediction once
    if args.debug:
        print(f"Predicting with {num_values} historical prices...")

    try:
        # Adjusted ARIMA order based on common practice for price data
        model = pmdarima.ARIMA(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), maxiter=10000)
        model = model.fit(ts)
        # print(model.summary())
        # print("Best order found:", model.order)

        if args.print_all:
            forecast, conf_int = model.predict(n_periods=N_FORECASTS, return_conf_int=True)
        else:
            forecast = model.predict(n_periods=N_FORECASTS)

    except pmdarima.arima.utils.ModelFitWarning as e:
        print(f"⚠️ Model fit warning: {e}")
        # No exit(), but only if it doesn't block the forecast
    except Exception as e:
        print(f"❌ There was an error with the prediction: {e}")
        exit()

    # print(forecast) # Line for testing purposes only
    # np.set_printoptions(precision=12) # Line for testing purposes only
    forecast = np.array(forecast)
    print("Prediction done.")
    # print(forecast) # Line for testing purposes only
    # print(forecast[1]) # Line for testing purposes only

    # Find the maximum/minimum predicted value
    max_value = np.max(forecast)
    min_value = np.min(forecast)
    # print(f"max value: {max_value}") # Line for testing purposes only
    # min_value = np.min(forecast) # Line for testing purposes only

    max_index = np.argmax(forecast)
    min_index = np.argmin(forecast)
    # reverse_max = (forecast)[max_index] # Line for testing purposes only
    # min_index = (list(forecast).index(min_value)) # Line for testing purposes only
    # print(f"reverse max index: {reverse_max}") # Line for testing purposes only
    # print(f"min: {min_index}") # Line for testing purposes only
    # print(f"position {max_index}") # Line for testing purposes only
    # print(f"maximum {forecast[max_index]}") # Line for testing purposes only
    # print(f" first 10: {forecast[:10]}")  # first 10 # Line for testing purposes only
    # print(f"last 10: {forecast[-10:]}") # Line for testing purposes only
    # print(f" has different values (how many?)? np.unique(forecast)")  # how many different values?  # last 10 # Line for testing purposes only
    # print(f"is it a valid array? {type(forecast)}") # Line for testing purposes only
    # print(f"{forecast.shape}") # Line for testing purposes only

    try:
        # Calculate time for max/min prediction
        if buy_or_sell == "buyPrice":
            hours_missing = (max_index * interval_seconds) / 3600
            if hours_missing < 1:
                minutes_missing = (max_index * interval_seconds) / 60
                seconds_missing = round((minutes_missing - math.floor(minutes_missing)) * 60)
                minutes_missing = math.floor(minutes_missing)
            else:
                minutes_missing = round((hours_missing - math.floor(hours_missing)) * 60)
                seconds_missing = round((minutes_missing - math.floor(minutes_missing)) * 60) if minutes_missing > 0 else 0
                minutes_missing = math.floor(minutes_missing)
                hours_missing = math.floor(hours_missing)

            print(f"\nHighest predicted price: {round(max_value, 3)} coins")
            print(f"Expected in {hours_missing} Hours {minutes_missing} Minutes {seconds_missing} Seconds")

        elif buy_or_sell == "sellPrice":
            hours_missing = (min_index * interval_seconds) / 3600
            if hours_missing < 1:
                minutes_missing = (min_index * interval_seconds) / 60
                seconds_missing = round((minutes_missing - math.floor(minutes_missing)) * 60)
                minutes_missing = math.floor(minutes_missing)
            else:
                minutes_missing = round((hours_missing - math.floor(hours_missing)) * 60)
                seconds_missing = round((minutes_missing - math.floor(minutes_missing)) * 60) if minutes_missing > 0 else 0
                minutes_missing = math.floor(minutes_missing)
                hours_missing = math.floor(hours_missing)

            print(f"\nLowest predicted price: {round(min_value, 3)} coins")
            print(f"Expected in {hours_missing} Hours {minutes_missing} Minutes {seconds_missing} Seconds")

        print("You can stop this program by pressing the keys CTRL and C together.")

    except Exception as e:
        print(f"❌ There was an error whilst calculating the remaining time: {e}")
        exit()

    # print(f"📍 Step: {max_index + 1}") #
    # print(f" in about {round(hours_missing, 0)} Hours {round(minutes_missing)} Minutes {seconds_missing} Seconds")
    # print(model.order) # Line for testing purposes only
    # print(f"n of values: {num_values}") # Line for testing purposes only

    # Print all predictions if requested
    if args.print_all:
        print_anyway = input(f"Do you want to print every single prediction ({N_FORECASTS} predictions) [Y/N]? It might take some time to print (and to read): ").strip().lower()
        if print_anyway in ("y", "yes"):
            # print("📈 First ten predicted steps (± % confidence):") # Line for testing purposes only
            for i in range(N_FORECASTS):
                pred = forecast[i]
                if 'conf_int' in locals(): # Check if confidence intervals were calculated
                    lower, upper = conf_int[i]
                    max_deviation = max(abs(upper - pred), abs(pred - lower))
                    perc_deviation = (max_deviation / pred) * 100 if pred != 0 else 0
                    # print(f"Step {i+1}: {round(pred, 3)} ±{round(perc_deviation, 4)}% (≈ {round(max_deviation, 2)})") # Line for testing purposes only
                    print(f"{round(pred, 3)} ±{round(perc_deviation, 4)}%")
                else:
                    print(f"{round(pred, 3)}") # Print prediction without confidence if not available
            exit()
        elif print_anyway in ("n", "no"):
            exit()
        else:
            print("Please enter a valid answer (Y/N).")

    # Variables for comparison (debug mode)
    step = 0
    actual_values = []
    predicted_values = []

    # Debug comparison loop (currently commented out in original logic)
    # if args.debug:
        # print("    step    |  prediction   |  actual value | error (%) | conf (±val |  ±%)")
    # while step < N_FORECASTS: # This loop seems incomplete/unused in the original
    #     try:
    #         # now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    #         # url = "https://api.hypixel.net/v2/skyblock/bazaar"
    #         # response = requests.get(url, headers={"API-Key": api_key})
    #         # if response.status_code == 200:
    #         #     data = response.json()
    #         #     product = data.get("products", {}).get(product_id)
    #         #     if product:
    #         #         q = product.get("quick_status", {})
    #         #         sell_price = q.get("sellPrice", 0) # This logic needs fixing for buy/sell
    #         #         timestamp_utc = datetime.now(timezone.utc).isoformat()
    #         #         # Save new record
    #         #         # record = {
    #         #         #     "timestamp_utc": timestamp_utc,
    #         #         #     "timestamp_ms": now_ms,
    #         #         #     "product_id": product_id,
    #         #         #     "sellPrice": sell_price, # Needs adjustment for buy/sell
    #         #         # }
    #         #         # with open(output_file, "a") as f:
    #         #         #     f.write(json.dumps(record) + "\n")
    #         #         # Compare with corresponding forecast
    #         #         if args.debug:
    #         #             try:
    #         #                 pred = forecast[step]
    #         #                 actual = sell_price # Needs adjustment
    #         #                 err = percent_error(pred, actual) # Use translated function
    #         #                 # Confidence for this step
    #         #                 if 'conf_int' in locals():
    #         #                     lower, upper = conf_int[step]
    #         #                     max_dev = max(abs(upper - pred), abs(pred - lower))
    #         #                     perc_dev = (max_dev / pred) * 100 if pred != 0 else 0
    #         #                     predicted_values.append(pred)
    #         #                     actual_values.append(actual)
    #         #                     print(f"step: {step+1:<5} | {pred:.4f} | {actual:.4f} | {err:.6f}% | ±{max_dev:.2f}  |  ±{perc_dev:.4f}% ") if err is not None else \
    #         #                         print(f"step: {step+1:<5} | {pred:.4f} | {actual:.4f} | {'n.a.':>7} |  ±{max_dev:.2f} | ±{perc_dev:.4f}%")
    #         #             except IndexError as e:
    #         #                 print(f"❌ Forecast index out of range: {e}")
    #         #                 exit()
    #         #             except Exception as e:
    #         #                 print(f"❌ There was an error whilst calculating the debug values: {e}")
    #         #                 exit()
    #         #         step += 1
    #         #     else:
    #         #         print("Item not found")
    #         # else:
    #         #     print(f"❌ HTTP {response.status_code}")
    #     except Exception as e:
    #         break # print(f"❗ Error: {repr(e)}")
    #     time.sleep(interval_seconds)

    # print("✅ End of the 100 steps planned.") # Line for testing purposes only

except KeyboardInterrupt:
    print(f"[{datetime.utcnow().isoformat()}] 🛑 Script stopped by user.")
    exit()
except Exception as e:
    print(f"❌ Unexpected error occurred: {e}")
    exit()
