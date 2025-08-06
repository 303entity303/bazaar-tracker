# bazaar-tracker

# Hypixel Bazaar Price Predictor 🧮📈

This Python project predicts future Hypixel Bazaar prices based on historical data collected from the official API.  
It includes two scripts:

- `fetch.py`: fetches current Bazaar prices every 20 seconds and stores them in the `prices/` folder
- `predict.py`: uses the collected data to forecast future item prices

The script is designed to be easy to use even for non-expert users — just follow the setup instructions and you're ready to go!

## 🛠 Requirements
- Python 3.9+
- A valid Hypixel API key (tutorial available on my [YouTube channel](https://youtube.com/@303Entity303))

## 🔐 API Key
When first run, the script will ask for your Hypixel API key and save it locally.  
The key is used only to fetch public Bazaar data.

## 📁 Data Storage
The price history is saved locally, and depending on how long you leave the script running, it may use several hundred MB to a few GB of disk space per day.

## 📌 License
This project is licensed under the **CC BY-ND 4.0** license.  
You are free to use and share it, but **you may not modify or redistribute modified versions**.

For any questions or issues, contact **303Entity303**
