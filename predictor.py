import os
import json
#CONFIG_FILE = "./config.json"


    # Se esiste config.json, carica la chiave
#def load_api_key():
#    # Se esiste config.json, carica la chiave
#    if os.path.exists(CONFIG_FILE):
#        with open(CONFIG_FILE, "r") as f:
#            try:
#                data = json.load(f)
#                return data.get("api_key")
#            except json.JSONDecodeError as e:
#                print(f"config.json is invalid or corrupted: {e}")  # File corrotto o vuoto
#
#    # Se non esiste o non valida, chiedi all'utente
#    api_key = input("Insert your API KEY: ").strip()
#    with open(CONFIG_FILE, "w") as f:
#        json.dump({"api_key": api_key}, f)
#        print("to start predicting restart this program")
#        exit()
#    return api_key
#api_key = load_api_key()
import warnings
import argparse
import sys
import subprocess
import psutil
import signal
import platform
#====================================================================CONFIGURAZIONE PARAMETRI===================================================================
parser = argparse.ArgumentParser(description="Predict the future buy or sell price of an item on Hypixel Bazaar.")

parser.add_argument("mode", nargs="?", choices=["buy", "sell"], help="Select the type of prediction: 'buy' for buy price, 'sell' for sell price.")

parser.add_argument("item_name", nargs="?", help="The name of the item you want to predict the price for (e.g., ENCHANTED_DIAMOND).")

parser.add_argument("time", nargs="?", type=int, help="How far into the future you want the prediction to go (numeric value only, without unit).")

parser.add_argument("time_unit", nargs="?", choices=["s", "m", "h", "d"], help="Time unit for the prediction: 's' = seconds, 'm' for minutes, 'h' = hours, 'd' = days.")

parser.add_argument("--debug", action="store_true", help="Show detailed information about prediction accuracy and internals.")

parser.add_argument('--print-all', action='store_true', help='Print all predicted values instead of only the best one.')

parser.add_argument('--stop_fetcher', action='store_true', help='stops the fetcher script and exits')

args = parser.parse_args()

#========================================================================== importation ========================================================================

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

#========================================================================== warning skipping ========================================================================

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
#========================================================================== fetcher handling ========================================================================
PID_FILE = "fetcher.pid"
if os.path.exists(PID_FILE):
    fetcher_exist = True
else:
    fetcher_exist= False
if args.stop_fetcher:
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())

            proc = psutil.Process(pid)
            proc.terminate()  # Usa proc.kill() se vuoi forzare
            print(f"✅ process fetcher.py (PID {pid}) terminated.")
        except FileNotFoundError as e:
                print(f"ℹ️ fetcher.pid not found: {e}")
                # no exit(), viene già gestito sotto
        except ValueError as e:
            print(f"❌ Invalid PID in fetcher.pid: {e}")
            exit()
        except psutil.NoSuchProcess as e:
            print(f"⚠️ No such process with PID in fetcher.pid: {e}")
            # no exit(), ma si può continuare
        except psutil.AccessDenied as e:
            print(f"❌ Access denied when trying to terminate fetcher process: {e}")
            exit()
        finally:
            os.remove(PID_FILE)
    else:
        print("ℹ️ no file fetcher.pid found, nothing to kill.")
    exit()

FETCH_SCRIPT = "fetcher.py"
  # File per salvare il PID del processo fetcher.py
# Argomenti platform-specific
if not fetcher_exist:
    kwargs = {}
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

# Avvia fetcher.py
    proc = subprocess.Popen([sys.executable, FETCH_SCRIPT], **kwargs)

# Salva il PID su file
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))

        print(f"fetcher.py started")

#========================================================================= what to predict =========================================================================

if args.mode == "buy":
    buy_o_sell = "buyPrice"
elif args.mode == "sell":
    buy_o_sell = "sellPrice"

#============================================================================ item ============================================================================

cartella = "prices/"

oggetto = args.item_name
try:
    if not os.path.exists(cartella):
        os.makedirs(cartella)
except PermissionError as e:
    print(f"❌ Permission denied: cannot create directory '{cartella}'. {e}")
    exit()
except OSError as e:
    print(f"❌ Failed to create directory '{cartella}': {e}")
    exit()

trovato = False

for nome_file in os.listdir(cartella):
    if nome_file == f"{oggetto}.jsonl":
        trovato = True
        fileoggetto = f"{oggetto}.jsonl"

if trovato == False:
    print("ERROR: item not found, choose an existing one OR use 'python3 predictor.py -h' for help")
    exit()

#=============================================================================TEMPO=============================================================================

temporaw = args.time
unitatempo = args.time_unit

if unitatempo == "s":
    secondi = temporaw
elif unitatempo == "m":
    secondi = temporaw * 60
elif unitatempo == "h":
    secondi = temporaw * 3600
elif unitatempo == "d":
    secondi = temporaw * 86400
else:
    raise ValueError("time unit not valid. use 's', 'h' or 'd'.")

steps = round(secondi / 20)

#print(f"{temporaw}{unitatempo} = {secondi} secondi")
#print(f"Risultato (secondi / 20): {steps}")

try:
    #print("Continua con la predizione...")
    ## Configurazione
    getcontext().prec = 50
    product_id = oggetto
    intervallo_secondi = 20
    N_PREVISIONI = steps
    valorinu = 0
    input_file = f"prices/{product_id}.jsonl"
    #file_path = f"prices/{input_file}"
    def carica_dati(input_file):
        global valorinu
        prices, dates = [], []
        try:
            with open(input_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        price = data.get(buy_o_sell)
                        timestamp_ms = data.get("timestamp_ms")
                        if price is not None and timestamp_ms is not None:
                            prices.append(float(price))
                            dates.append(datetime.fromtimestamp(timestamp_ms / 1000))
                            valorinu += 1
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Skipping invalid JSON line: {e}")
        # no exit(), si può continuare
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

    def errore_percentuale(prev, real):
        if real == 0:
            return None
        return round(abs(prev - real) / real * 100, 5)

    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # Carica dati storici
    ts = carica_dati(input_file)

    if len(ts) < 10:
        print("\033[91mERROR\033[0m not enough data, recommended at least 40 values(about 13 minutes worth of data collection)")
        exit()

    if len(ts) < 40:
        print("\033[93mWARNING\33[0, i personally recommend at least 40 values(about 13 minutes worth of data collection to predict about 30 miutes)")
        exit()

    # Previsione una sola volta
    if args.debug:
         print(f"predicting with {valorinu} historycal prices...")
    try:
        model = pmdarima.ARIMA(order=(0, 1, 2), maxiter=10000)
        model = model.fit(ts)

        #print(model.summary())
        #print("Best order found:", model.order)
        if args.print_all:
            forecast, conf_int = model.predict(n_periods=N_PREVISIONI, return_conf_int=True)
        else:
            forecast= model.predict(n_periods=N_PREVISIONI)
    except pmdarima.arima.utils.ModelFitWarning as e:
        print(f"⚠️ Model fit warning: {e}")
        # no exit(), ma solo se non blocca il forecast
    except Exception as e:
        print(f"❌ There was an error with the prediction:\n{e}")
        exit()
#    print(forecast) this line has no purpose and has only been used for testing purposes
#    np.set_printoptions(precision=12) this line has no purpose and has only been used for testing purposes
    forecast = np.array(forecast)
    print("prediction done")
#    print(forecast) this line has no purpose and has only been used for testing purposes
#    print(forecast[1]) this line has no purpose and has only been used for testing purposes

    # Trova il massimo previsto
    max_value = np.max(forecast)
    min_value = np.min(forecast)
#    print(f"max value: {max_value}") this line has no purpose and has only been used for testing purposes
#    min_value = np.min(forecast) this line has no purpose and has only been used for testing purposes

    max_index = np.argmax(forecast)
    min_index = np.argmin(forecast)
#    reverse_max = (forecast)[max_index] this line has no purpose and has only been used for testing purposes
#    min_index = (list(forecast).index(min_value)) this line has no purpose and has only been used for testing purposes
#    print(f"reverse max index: {reverse_max}") this line has no purpose and has only been used for testing purposes
#    print(f"min: {min_index}") this line has no purpose and has only been used for testing purposes
#    print(f"posizione {max_index}") this line has no purpose and has only been used for testing purposes
#    print(f"massimo {forecast[max_index]}")this line has no purpose and has only been used for testing purposes
#    print(f" primi 10: {forecast[:10]}")  # primi 10 this line has no purpose and has only been used for testing purposes
#    print(f"ultimi 10: {forecast[-10:]}") this line has no purpose and has only been used for testing purposes
#    print(f" ha valori diversi(quanti?)? np.unique(forecast)")  # quanti valori diversi ha?  # ultimi 10 #this line has no purpose and has only been used for testing purposes
#    print(f"è un array valido? {type(forecast)}") this line has no purpose and has only been used for testing purposes
#    print(f"{forecast.shape}") this line has no purpose and has only been used for testing purposes
    try:
        if buy_o_sell == "buyPrice":
            ore_mancanti = (max_index * intervallo_secondi) / 3600

            if (math.floor(ore_mancanti)) < 1:
                minuti_mancanti = (max_index * intervallo_secondi) / 60

            elif (math.floor(ore_mancanti)) >= 1:
                minuti_mancanti = round(ore_mancanti - math.floor(ore_mancanti)) * 60

                secondi_mancanti =round(minuti_mancanti - math.floor(minuti_mancanti)) * 60
            print(f"\n Highest predicted price: {round(max_value, 3)} coins")
            print(f" Expected in {round(ore_mancanti, 0)} Hours {round(minuti_mancanti)} Minutes {secondi_mancanti} Seconds")
        elif buy_o_sell == "sellPrice":
            ore_mancanti = (min_index * intervallo_secondi) / 3600

            if (math.floor(ore_mancanti)) < 1:
                minuti_mancanti = (min_index * intervallo_secondi) / 60
                secondi_mancanti =round(minuti_mancanti - math.floor(minuti_mancanti)) * 60

            elif (math.floor(ore_mancanti)) >= 1:
                minuti_mancanti = round(ore_mancanti - math.floor(ore_mancanti)) * 60

                secondi_mancanti =round(minuti_mancanti - math.floor(minuti_mancanti)) * 60
        print(f"\n Lowest predicted price: {round(min_value, 3)} coins")
        print(f" Expected in {round(ore_mancanti, 0)} Hours {round(minuti_mancanti)} Minutes {secondi_mancanti} Seconds")
        print ("You can stop this program by pressing the keys CTRL and C together.")
    except Exception as e:
        print(f"❌ There was an error whilst calculating the remaining time:\n{e}")
        exit()
    #print(f"📍 Step: {max_index + 1}") #
    #print(f" in about {round(ore_mancanti, 0)} Hours {round(minuti_mancanti)} Minutes {secondi_mancanti} Seconds")
    #print(model.order) #this line has no purpose and has only been used for testing purposes

    #print(f"n di valori: {valorinu}") #this line has no purpose and has only been used for testing purposes
#        print("📈 first ten predicted steps (± % confidenza):") this line has no purpose and has only been used for testing purposes
    if args.print_all:
        print_anyway = input(f"do you want to print every single prediction() ({N_PREVISIONI} predictions) [Y/N]\n it might take some time to print (and to read)").strip().lower()
        if print_anyway in ("y", "Y", "yes", "YES"):
            for i in range(N_PREVISIONI): #this line has no purpose and has only been used for testing purposes
                pred = forecast[i] #this line has no purpose and has only been used for testing purposes
                lower, upper = conf_int[i] #this line has no purpose and has only been used for testing purposes

                max_deviation = max(abs(upper - pred), abs(pred - lower)) #this line has no purpose and has only been used for testing purposes
                perc_deviation = (max_deviation / pred) * 100 if pred != 0 else 0 #this line has no purpose and has only been used for testing purposes

                #print(f"Step {i+1}: {round(pred, 3)} ±{round(perc_deviation, 4)}% (≈ {round(max_deviation, 2)})") #this line has no purpose and has only been used for testing purposes
                print(f"{round(pred, 3)} ±{round(perc_deviation, 4)}%") #this line has no purpose and has only been used for testing purposes
            exit()
        elif print_anyway in ("n", "N", "no", "NO"):
            exit()
        else:
            print("enter a valid answer")

    # Variabili per confronto
    step = 0
    actual_values = []
    predicted_values = []
    #if args.debug:
        #print("    step    |  prediction   |  actual value | error (%) | conf (±val |  ±%)")

    while step < N_PREVISIONI:
        try:
            #now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            #url = "https://api.hypixel.net/v2/skyblock/bazaar"
            #response = requests.get(url, headers={"API-Key": api_key})

            #if response.status_code == 200:
                #data = response.json()
                #prodotto = data.get("products", {}).get(product_id)

                #if prodotto:
                    #q = prodotto.get("quick_status", {})
                    #sell_price =q.get("sellPrice", 0)
                    #timestamp_utc = datetime.now(timezone.utc).isoformat()

                    # Salva nuovo record
                    #record = {
                        #"timestamp_utc": timestamp_utc,
                        #"timestamp_ms": now_ms,
                        #"product_id": product_id,
                        #"sellPrice": sell_price,
                    #}
                    #with open(output_file, "a") as f:
                    #    f.write(json.dumps(record) + "\n")

                    # Confronta con previsione corrispondente
                    if args.debug:
                        try:
                            pred = forecast[step]
                            actual = sell_price
                            err = errore_percentuale(pred, actual)

                            # Confidenza per questo step
                            lower, upper = conf_int[step]
                            max_dev = max(abs(upper - pred), abs(pred - lower))
                            perc_dev = (max_dev / pred) * 100 if pred != 0 else 0

                            predicted_values.append(pred)
                            actual_values.append(actual)
                            print(f"step: {step+1:<5} | {pred:.4f} | {actual:.4f} | {err:.6f}% | ±{max_dev:.2f}  |  ±{perc_dev:.4f}% ") if err is not None else \
                                print(f"step: {step+1:<5} | {pred:.4f} | {actual:.4f} | {'n.a.':>7} |  ±{max_dev:.2f} | ±{perc_dev:.4f}%")
                        except IndexError as e:
                            print(f"❌ Forecast index out of range: {e}")
                            exit()
                        except Exception as e:
                            print(f"❌ There was an error whilst calculating the debug values:\n{e}")
                            exit()
                    step += 1
                #else:
                #    print("item not found")
            #else:
            #    print(f"❌ HTTP {response.status_code}")

        except Exception as e:
            break #print(f"❗ Errore: {repr(e)}")
        time.sleep(intervallo_secondi)
#    print("✅ Fine dei 100 step previsti.") this line has no purpose and has only been used for testing purposes
except KeyboardInterrupt:
    exit()
except Exception as e:
    print(f"❌ Unexpected error occurred: {e}")
    exit()
