import json

import fastapi
from exchange_scrapper import scan_market, get_available_exhanges, get_all_pairs, scan_single_coin
import os
app = fastapi.FastAPI()

@app.get("/scan_market/{timeframe}/{exchange_name}")
def api_scan_market(timeframe: str, exchange_name: str):
    exchange_name = exchange_name.lower()
    if timeframe not in ["5m","15m","1h"]:
        raise fastapi.HTTPException(status_code=400, detail="timeframe must be 5m, 15m or 1h")
        #return {"error":"timeframe must be 5m, 15m or 1h"}
    try:
        scan_market(timeframe,exchange_name)
        if os.path.exists("live_market_data.json"):
            with open("live_market_data.json", "r",encoding='utf-8') as f:
                data = json.load(f)
                return data
        else:
            raise fastapi.HTTPException(status_code=404, detail="live_market_data.json not found.But scan completed successfully.")
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e)+" tarama işlemi başlatılamadı")

@app.get("/get_last_data")
def api_get_last_data():
    last_update_time = 0
    if os.path.exists("live_market_data.json"):
        with open("live_market_data.json", "r",encoding='utf-8') as f:
            data = json.load(f)
            first_coin_key = next(iter(data))
            last_update_time = data[first_coin_key].get("updated_at")
            response = {"last_update_time":last_update_time,
                        "data":data}
            return response
    else:
        return {"error":"market data not found."}

@app.get("/get_coin_data/{coin_name}")
def api_get_coin_data(coin_name: str):
    coin_name = coin_name.upper()
    if os.path.exists("live_market_data.json"):
        with open("live_market_data.json","r",encoding='utf-8') as f:
            data = json.load(f)
            coin_data = data.get(f"{coin_name}/USDT")
            return coin_data

@app.get("/scan_coin/{timeframe}/{exchange_name}/{coin_name}")
def api_scan_coin(timeframe:str,exchange_name:str,coin_name: str):
    coin_name = coin_name.upper()
    exchange_name = exchange_name.lower()
    available_coins = get_all_pairs(exchange_name)
    if f"{coin_name}/USDT" not in available_coins:
        raise fastapi.HTTPException(status_code=404, detail=f"{coin_name} not found in {exchange_name} exchange.")
    else:
        return scan_single_coin(coin_name, timeframe, exchange_name)