using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;

// Python'dan gelen veriyi temsil eden model
public class MarketDataResponse
{
    // JSON yapınıza göre bu alanları özelleştirebiliriz.
    // Örnek olarak dinamik bir Dictionary kullanıyorum.
    public Dictionary<string, CoinInfo> Data { get; set; }
}

public class CoinInfo
{
    // Python tarafındaki JSON içeriğine göre buraları doldurmalısınız
    public decimal Price { get; set; }
    public decimal Volume { get; set; }
    public string UpdatedAt { get; set; }
}