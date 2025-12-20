using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace CRYPTAI.Services
{
    public class FavoritesService
    {
        private const string FavFileName = "favorites.json";

        // Hızlı erişim için HashSet kullanıyoruz (O(1) karmaşıklığı)
        private HashSet<string> _favoriteSymbols = new();

        // Dosya çakışmalarını önlemek için kilit objesi
        private readonly object _fileLock = new object();

        public void Load()
        {
            try
            {
                string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, FavFileName);

                // Dosya okuma işlemi
                lock (_fileLock)
                {
                    if (File.Exists(path))
                    {
                        string json = File.ReadAllText(path);
                        var list = JsonSerializer.Deserialize<List<string>>(json);
                        _favoriteSymbols = new HashSet<string>(list ?? new List<string>());
                    }
                }
            }
            catch
            {
                // Okuma hatası olursa boş liste ile devam et, sistemi çökertme
                _favoriteSymbols = new HashSet<string>();
            }
        }

        public void Save()
        {
            try
            {
                string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, FavFileName);

                // Dosya yazarken başka thread'in girmesini engelle
                lock (_fileLock)
                {
                    string json = JsonSerializer.Serialize(_favoriteSymbols.ToList());
                    File.WriteAllText(path, json);
                }
            }
            catch
            {
                // Kayıt hatası loglanabilir
            }
        }

        public bool IsFavorite(string symbol)
        {
            return _favoriteSymbols.Contains(symbol);
        }

        public void ToggleFavorite(string symbol)
        {
            if (_favoriteSymbols.Contains(symbol))
            {
                _favoriteSymbols.Remove(symbol);
            }
            else
            {
                _favoriteSymbols.Add(symbol);
            }

            // Her değişiklikte diske yaz (Kalıcılık)
            Save();
        }
    }
}