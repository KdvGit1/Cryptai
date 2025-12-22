using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows.Media;
using CRYPTAI.Models;
using Brushes = System.Windows.Media.Brushes;

namespace CRYPTAI.Services
{
    public class MarketDataService
    {
        private readonly HttpClient _httpClient = new();
        private const string ApiUrl = "http://127.0.0.1:8000/get_last_data";
        private const string ScanUrl = "http://127.0.0.1:8000/scan_market/15m/binance";

        // Parametre olarak gelen FavoritesService kaldırıldı.
        public async Task<List<CoinItem>> GetMarketDataAsync()
        {
            var response = await _httpClient.GetStringAsync(ApiUrl);
            using JsonDocument doc = JsonDocument.Parse(response);
            var root = doc.RootElement;
            var coinList = new List<CoinItem>();

            if (root.TryGetProperty("data", out var dataElement))
            {
                foreach (var property in dataElement.EnumerateObject())
                {
                    var symbol = property.Name;
                    var coinDetails = property.Value;

                    decimal price = 0;
                    decimal volume = 0;
                    decimal atr = 0;
                    decimal aiRate = 0;

                    if (coinDetails.TryGetProperty("last_indicators", out var indicatorsArray)
                        && indicatorsArray.ValueKind == JsonValueKind.Array)
                    {
                        var items = new List<JsonElement>();
                        foreach (var item in indicatorsArray.EnumerateArray()) items.Add(item);

                        if (items.Count > 0)
                        {
                            var latestData = items.Last();
                            if (latestData.TryGetProperty("Close", out var c)) price = c.GetDecimal();
                            if (latestData.TryGetProperty("Volume", out var v)) volume = v.GetDecimal();
                            if (latestData.TryGetProperty("ATR_Val", out var a)) atr = a.GetDecimal();
                        }
                    }

                    if (coinDetails.TryGetProperty("ai_prediction", out var aiElement))
                        aiRate = aiElement.GetDecimal();

                    decimal expectedPrice = price + (price * aiRate);

                    string signalType = "NÖTR ➖";
                    SolidColorBrush signalColor = Brushes.Gray;
                    if (aiRate > 0) { signalType = "LONG 🚀"; signalColor = Brushes.LimeGreen; }
                    else if (aiRate < 0) { signalType = "SHORT 🔻"; signalColor = Brushes.Red; }

                    decimal volumeRisk = 0;
                    if (volume < 50000) volumeRisk = 100;
                    else if (volume < 500000) volumeRisk = 70;
                    else if (volume < 5000000) volumeRisk = 40;
                    else volumeRisk = 10;

                    decimal volatilityRisk = 0;
                    if (price > 0)
                    {
                        decimal volatilityPercent = (atr / price) * 100;
                        if (volatilityPercent > 5) volatilityRisk = 100;
                        else if (volatilityPercent > 3) volatilityRisk = 70;
                        else if (volatilityPercent > 1) volatilityRisk = 30;
                        else volatilityRisk = 10;
                    }

                    decimal totalRiskScore = (volumeRisk + volatilityRisk) / 2;
                    string riskRatioText = $"%{totalRiskScore:0}";
                    SolidColorBrush riskColor = Brushes.White;

                    if (totalRiskScore >= 70) riskColor = Brushes.Red;
                    else if (totalRiskScore >= 40) riskColor = Brushes.Orange;
                    else riskColor = Brushes.Cyan;
                    coinList.Add(new CoinItem
                    {
                        Symbol = symbol,
                        Price = price,
                        Volume = volume,
                        AiPercentage = aiRate,
                        ExpectedPrice = expectedPrice,
                        SignalType = signalType,
                        SignalColor = signalColor,
                        RiskRatio = riskRatioText,
                        RiskColor = riskColor
                    });
                }
            }
            return coinList;
        }

        public async Task TriggerScanAsync()
        {
            using (var scanClient = new HttpClient())
            {
                // Zaman aşımını koruyoruz
                scanClient.Timeout = TimeSpan.FromMinutes(20);

                // İsteği gönderiyoruz
                var response = await scanClient.GetAsync(ScanUrl);

                // KRİTİK HAMLE 1: Eğer sunucu 200 OK dışında bir şey (404, 500 vs.) dönerse
                // bu satır kodun akışını keser ve HATA fırlatır (Exception).
                // Böylece MainWindow "Başarılı" mesajı göstermez, Catch bloğuna düşer.
                response.EnsureSuccessStatusCode();

                // KRİTİK HAMLE 2: Sunucudan gelen mesajı okuyoruz (İsteğe bağlı loglama için)
                string responseContent = await response.Content.ReadAsStringAsync();

                // Debug için çıktı penceresine yazdıralım
                System.Diagnostics.Debug.WriteLine($"SCAN RESPONSE: {responseContent}");
            }
        }
    }
}