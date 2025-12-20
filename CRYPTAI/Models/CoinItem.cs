using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Media;
using Brushes = System.Windows.Media.Brushes;

namespace CRYPTAI.Models
{
    public class CoinItem : INotifyPropertyChanged
    {
        public string Symbol { get; set; } = string.Empty;
        public decimal Price { get; set; }
        public decimal Volume { get; set; }
        public decimal AiPercentage { get; set; }
        public decimal ExpectedPrice { get; set; }
        public string SignalType { get; set; } = string.Empty;
        public SolidColorBrush SignalColor { get; set; } = Brushes.White;
        public string RiskRatio { get; set; } = "N/A";
        public SolidColorBrush RiskColor { get; set; } = Brushes.White;

     
        public event PropertyChangedEventHandler? PropertyChanged;
        protected void OnPropertyChanged([CallerMemberName] string? name = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }
}