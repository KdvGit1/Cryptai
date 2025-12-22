import json
import fastapi
import sys
import io
import os
import traceback  # <--- Hatanın röntgenini çekmek için ekledik

# exchange_scrapper içindeki hataları yakalamak için importu try-catch içine almıyoruz
# ama eğer burada hata varsa sunucu hiç açılmaz.
try:
    from exchange_scrapper import scan_market, get_available_exhanges, get_all_pairs, scan_single_coin
except ImportError as ie:
    print(f"❌ MODÜL HATASI: exchange_scrapper veya bağımlılıkları yüklenemedi.\nDetay: {ie}")
    sys.exit(1)


app = fastapi.FastAPI()

@app.get("/scan_market/{timeframe}/{exchange_name}")
def api_scan_market(timeframe: str, exchange_name: str):
    exchange_name = exchange_name.lower()
    
    print(f"🚀 İSTEK GELDİ: Tarama Başlatılıyor... [{exchange_name} - {timeframe}]")

    if timeframe not in ["5m","15m","1h"]:
        raise fastapi.HTTPException(status_code=400, detail="Timeframe must be 5m, 15m or 1h")

    try:
        # Taramayı başlatıyoruz
        scan_market(timeframe, exchange_name)
        
        # Dosyayı kontrol ediyoruz
        if os.path.exists("live_market_data.json"):
            with open("live_market_data.json", "r", encoding='utf-8') as f:
                data = json.load(f)
                print("✅ Tarama Başarılı, veri dönülüyor.")
                return data
        else:
            print("⚠️ HATA: scan_market çalıştı ama JSON dosyası oluşmadı.")
            raise fastapi.HTTPException(status_code=404, detail="Scan completed but output file not found.")

    except Exception as e:
        # Hatanın tam dökümünü al (Hangi dosya, hangi satır)
        error_trace = traceback.format_exc()
        
        print("\n" + "="*40)
        print("❌ PYTHON TARAFINDA KRİTİK HATA!")
        print("="*40)
        print(error_trace)
        print("="*40 + "\n")
        
        # C# tarafına hatanın sadece başlığını değil, detayını gönderiyoruz
        # Böylece Alert Dialog'da hatanın tam sebebini göreceksin.
        raise fastapi.HTTPException(status_code=500, detail=f"BACKEND ERROR: {str(e)}\n\nTrace: {error_trace}")

@app.get("/get_last_data")
def api_get_last_data():
    try:
        if os.path.exists("live_market_data.json"):
            with open("live_market_data.json", "r", encoding='utf-8') as f:
                data = json.load(f)
                # Basit bir kontrol: data boş mu?
                if not data:
                    return {"error": "Data file is empty"}
                
                first_coin_key = next(iter(data))
                last_update_time = data[first_coin_key].get("updated_at", "N/A")
                response = {"last_update_time": last_update_time, "data": data}
                return response
        else:
            return {"error": "market data not found."}
    except Exception as e:
        print(f"❌ Veri okuma hatası: {e}")
        return {"error": str(e)}

@app.get("/get_coin_data/{coin_name}")
def api_get_coin_data(coin_name: str):
    coin_name = coin_name.upper()
    if os.path.exists("live_market_data.json"):
        with open("live_market_data.json", "r", encoding='utf-8') as f:
            data = json.load(f)
            # Slash sorunu yaşamamak için:
            key = f"{coin_name}/USDT"
            if key in data:
                return data[key]
            # Belki kullanıcı /USDT yazmıştır diye kontrol
            elif coin_name in data:
                return data[coin_name]
            else:
                 raise fastapi.HTTPException(status_code=404, detail="Coin not found")
    else:
        raise fastapi.HTTPException(status_code=404, detail="live_market_data.json not found.")

@app.get("/scan_coin/{timeframe}/{exchange_name}/{coin_name}")
def api_scan_coin(timeframe:str, exchange_name:str, coin_name: str):
    try:
        coin_name = coin_name.upper()
        exchange_name = exchange_name.lower()
        available_coins = get_all_pairs(exchange_name)
        
        target_pair = f"{coin_name}/USDT"
        if target_pair not in available_coins:
            raise fastapi.HTTPException(status_code=404, detail=f"{coin_name} not found in {exchange_name}.")
        else:
            return scan_single_coin(coin_name, timeframe, exchange_name)
    except Exception as e:
         raise fastapi.HTTPException(status_code=500, detail=str(e))

@app.get("/get_available_exchanges")
def api_get_available_exchanges():
    return get_available_exhanges()