import requests
import time
import json
import math
import os
from datetime import datetime, timezone
print("🛑 You can stop this program anytime using the command 'python3 predictor.py --stop_fetcher '")
print("📦 This script retrieves Bazaar item prices and saves them in the 'prices' folder.")
print("🔮 Note: This script only collects data — it does NOT make predictions.")

print("\n📈 To get accurate predictions later, it is strongly recommended to keep this script running continuously.")
print("💾 Warning: This script can generate from several hundred megabytes up to a few gigabytes of data per day.")
#CONFIG_FILE = "./config.json"
#def load_api_key():
#    # Se esiste config.json, carica la chiave
#    if os.path.exists(CONFIG_FILE):
#        with open(CONFIG_FILE, "r") as f:
#            try:
#                data = json.load(f)
#                return data.get("api_key")
#            except json.JSONDecodeError:
#                pass  # File corrotto o vuoto
## === CONFIGURAZIONE ===
#api_key = load_api_key()
intervallo_secondi = 20
output_dir = "prices"

# Crea cartella 'prices/' se non esiste
try:
    os.makedirs(output_dir, exist_ok=True)

        # Costanti SkyBlock calendario
    months = [
        'Early Spring', 'Spring', 'Late Spring',
        'Early Summer', 'Summer', 'Late Summer',
        'Early Autumn', 'Autumn', 'Late Autumn',
        'Early Winter', 'Winter', 'Late Winter'
    ]

    hour_ms = 50000
    day_ms = 24 * hour_ms
    month_length = 31
    month_ms = month_length * day_ms
    year_length = len(months)
    year_ms = year_length * month_ms
    year_zero = 1560275700000  # SkyBlock epoch in ms

    def time_to_skyblock_date(time_ms):
        offset = time_ms - year_zero
        year = math.floor(offset / year_ms) + 1
        offset %= year_ms

        month = math.floor(offset / month_ms)
        offset %= month_ms

        day = math.floor(offset / day_ms) + 1
        offset %= day_ms

        hour = math.floor(offset / hour_ms)
        minute = int(((offset % hour_ms) / hour_ms) * 60)

        suffix = "pm" if hour >= 12 else "am"
        hour_12 = hour
        if hour > 12:
            hour_12 = hour - 12
        elif hour == 0:
            hour_12 = 12

        time_str = f"{hour_12}:{(minute // 10) * 10:02d}{suffix}"

        return {
            "year": year,
            "month_index": month,
            "month_name": months[month],
            "day": day,
            "hour": hour,
            "minute": minute,
            "time_str": time_str,
        }

    # === LOOP RACCOLTA DATI PER TUTTI GLI ITEM ===
    while True:
        try:
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            url = "https://api.hypixel.net/v2/skyblock/bazaar"
            response = requests.get(url)#, headers={"API-Key": api_key})

            if response.status_code == 200:
                data = response.json()
                all_products = data.get("products", {})

                skyblock_date = time_to_skyblock_date(now_ms)

                for product_id, product_data in all_products.items():
                    quick_status = product_data.get("quick_status", {})

                    record = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "timestamp_ms": now_ms,
                        "product_id": product_id,
                        "buyPrice": round((quick_status.get("buyPrice")), 3) if quick_status.get("buyPrice") is not None else None,
                        "sellPrice": round((quick_status.get("sellPrice")), 3) if quick_status.get("sellPrice") is not None else None,
                        "buyOrders": quick_status.get("buyOrders"),
                        "sellOrders": quick_status.get("sellOrders"),
                        "skyblock_year": skyblock_date["year"],
                        "skyblock_month": skyblock_date["month_name"],
                        "skyblock_day": skyblock_date["day"],
                        "skyblock_hour": skyblock_date["hour"],
                        "skyblock_minute": skyblock_date["minute"],
                        "skyblock_time_str": skyblock_date["time_str"],
                    }

                    output_file = os.path.join(output_dir, f"{product_id}.jsonl")

                    with open(output_file, "a") as f:
                        f.write(json.dumps(record) + "\n")

                    #print(f"[{record['timestamp_utc']}] Salvato: {product_id} → buy {record['buyPrice']} / sell {record['sellPrice']}")

            else:
                print(f"[{datetime.utcnow().isoformat()}] ❌ Errore HTTP: {response.status_code}")

        except Exception as e:
            print(f"[{datetime.utcnow().isoformat()}] ❗ Eccezione: {e}")

        time.sleep(intervallo_secondi)
except KeyboardInterrupt:
    exit()
