using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

namespace CRYPTAI
{
    public class PythonMarketService
    {
        private readonly HttpClient _httpClient;
        private const string BaseUrl = "http://127.0.0.1:8000"; // Python servis adresi

        public PythonMarketService()
        {
            _httpClient = new HttpClient();
        }


        /// <summary>
        /// Piyasayı tarar ve güncel veriyi çeker.
        /// Endpoint: /scan_market/{timeframe}/{exchange_name}
        /// </summary>
        public async Task<string> ScanMarketAsync(string timeframe, string exchangeName)
        {
            try
            {
                // Endpoint inşası
                string endpoint = $"{BaseUrl}/scan_market/{timeframe}/{exchangeName}";

                // Asenkron GET isteği
                HttpResponseMessage response = await _httpClient.GetAsync(endpoint);

                // Hata kontrolü (200 OK dışındaki durumlar)
                if (!response.IsSuccessStatusCode)
                {
                    throw new Exception($"Servis Hatası: {response.StatusCode}");
                }

                // JSON verisini string olarak al
                string jsonResponse = await response.Content.ReadAsStringAsync();
                return jsonResponse;
            }
            catch (Exception ex)
            {
                // Loglama mekanizması buraya eklenebilir
                return $"Bağlantı Hatası: {ex.Message}";
            }
        }

        /// <summary>
        /// Sadece son veriyi çeker (Scan yapmadan).
        /// Endpoint: /get_last_data
        /// </summary>
        public async Task<string> GetLastDataAsync()
        {
            var response = await _httpClient.GetAsync($"{BaseUrl}/get_last_data");
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync();
        }
    }
}
