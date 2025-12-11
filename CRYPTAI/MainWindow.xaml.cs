using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;
using System.Windows.Controls.Primitives;
using Binding = System.Windows.Data.Binding;
using Brushes = System.Windows.Media.Brushes;
using Color = System.Windows.Media.Color;
using ColorConverter = System.Windows.Media.ColorConverter;
using Image = System.Windows.Controls.Image;
using MouseEventArgs = System.Windows.Input.MouseEventArgs;
using ScottPlot;
using ScottPlot.Plottables; // Bu kalsa da olur ama Bar için gerekmez

namespace CRYPTAI
{
    // --- YARDIMCI SINIF: Finansal Hesaplamalar ---
    public static class FinanceIndicators
    {
        public static double[] CalculateSMA(List<OHLC> candles, int period)
        {
            double[] sma = new double[candles.Count];
            for (int i = 0; i < candles.Count; i++)
            {
                if (i < period - 1)
                {
                    sma[i] = double.NaN; // Yeterli veri yok
                }
                else
                {
                    double sum = 0;
                    for (int j = 0; j < period; j++)
                    {
                        sum += candles[i - j].Close;
                    }
                    sma[i] = sum / period;
                }
            }
            return sma;
        }
    }

    public partial class MainWindow : Window
    {
        private readonly HttpClient _httpClient = new();
        private const string ApiUrl = "http://127.0.0.1:8000/get_last_data";
        private const string ScanUrl = "http://127.0.0.1:8000/scan_market/15m/binance";
        private System.Diagnostics.Process? _pythonProcess;

        private List<CoinItem> _allCoinsBackup = new();

        public MainWindow()
        {
            InitializeComponent();
            SetupDataGrids();

            FavoriteText.Opacity = 0;
            HomePAGEText.Opacity = 0;

            StartPythonBackend();

            this.Closing += MainWindow_Closing;
            System.Windows.Application.Current.Exit += (s, e) => KillPythonProcess();
        }

        private async void MainWin_Loaded(object sender, RoutedEventArgs e)
        {
            Sagpanel.Width = new GridLength(9.2, GridUnitType.Star);
            Solpanel.Width = new GridLength(0.8, GridUnitType.Star);

            if (LoadingGrid is not null) LoadingGrid.Visibility = Visibility.Visible;
            await LoadMarketDataAsync(retryUntilSuccess: true);
            if (LoadingGrid is not null) LoadingGrid.Visibility = Visibility.Collapsed;
        }

        private async void MyDataGrid_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (MyDataGrid.SelectedItem is CoinItem selectedCoin)
            {
                await LoadAndShowCoinChart(selectedCoin.Symbol);
            }
        }

        private void BackToListViewButton_Click(object sender, RoutedEventArgs e)
        {
            if (ListAndTerminalContainer is not null) ListAndTerminalContainer.Visibility = Visibility.Visible;
            if (ChartContainer is not null) ChartContainer.Visibility = Visibility.Collapsed;
        }

        // --- GÜNCELLENMİŞ GRAFİK MOTORU (Fix: ScottPlot.Bar) ---
        // --- GÜNCELLENMİŞ GRAFİK MOTORU (Split View: Fiyat Üstte, Hacim Altta) ---
        private async Task LoadAndShowCoinChart(string symbol)
        {
            if (ListAndTerminalContainer is not null) ListAndTerminalContainer.Visibility = Visibility.Collapsed;
            if (ChartContainer is not null) ChartContainer.Visibility = Visibility.Visible;

            // --- ORTAK TEMA AYARLARI ---
            ScottPlot.Color bgColor = ScottPlot.Color.FromHex("#161A25");
            ScottPlot.Color gridMajor = ScottPlot.Color.FromHex("#2B2F36");
            ScottPlot.Color gridMinor = ScottPlot.Color.FromHex("#1E222D");
            ScottPlot.Color axisColor = ScottPlot.Color.FromHex("#848E9C");

            // 1. FİYAT GRAFİĞİ AYARLARI (ÜST)
            WpfPlotPrice.Plot.Clear();
            WpfPlotPrice.Plot.FigureBackground.Color = bgColor;
            WpfPlotPrice.Plot.DataBackground.Color = bgColor;
            WpfPlotPrice.Plot.Grid.MajorLineColor = gridMajor;
            WpfPlotPrice.Plot.Grid.MinorLineColor = gridMinor;
            WpfPlotPrice.Plot.Axes.Color(axisColor);

            // Sağ Eksen Aktif, Sol Pasif
            WpfPlotPrice.Plot.Axes.Left.IsVisible = false;
            WpfPlotPrice.Plot.Axes.Right.IsVisible = true;
            WpfPlotPrice.Plot.Axes.Right.TickLabelStyle.ForeColor = axisColor;
            // Alt eksendeki tarihleri gizle (Çünkü alttaki hacim grafiğinde görünecek)
            WpfPlotPrice.Plot.Axes.Bottom.TickLabelStyle.IsVisible = false;

            // 2. HACİM GRAFİĞİ AYARLARI (ALT)
            WpfPlotVolume.Plot.Clear();
            WpfPlotVolume.Plot.FigureBackground.Color = bgColor;
            WpfPlotVolume.Plot.DataBackground.Color = bgColor;
            WpfPlotVolume.Plot.Grid.MajorLineColor = gridMajor;
            WpfPlotVolume.Plot.Grid.MinorLineColor = gridMinor;
            WpfPlotVolume.Plot.Axes.Color(axisColor);

            // Sağ Eksen Aktif (Hacim Değerleri), Sol Pasif
            WpfPlotVolume.Plot.Axes.Left.IsVisible = false;
            WpfPlotVolume.Plot.Axes.Right.IsVisible = true;
            WpfPlotVolume.Plot.Axes.Right.TickLabelStyle.ForeColor = axisColor;
            // Alt eksen tarihleri görünsün
            WpfPlotVolume.Plot.Axes.DateTimeTicksBottom();

            // --- BAŞLIKLARI KALDIR ---
            WpfPlotPrice.Plot.Title("");
            WpfPlotVolume.Plot.Title("");

            WpfPlotPrice.Refresh();
            WpfPlotVolume.Refresh();

            AppendToTerminal($"Grafik verisi çekiliyor: {symbol}...");
            string apiSymbol = symbol.Replace("/USDT", "");

            try
            {
                var response = await _httpClient.GetStringAsync($"http://127.0.0.1:8000/get_coin_data/{apiSymbol}");
                using var jsonDoc = JsonDocument.Parse(response);

                if (jsonDoc.RootElement.ValueKind == JsonValueKind.Null || jsonDoc.RootElement.ValueKind != JsonValueKind.Object)
                {
                    WpfPlotPrice.Plot.Title("HATA: Veri Yok");
                    return;
                }

                // --- Verileri Parse Et ---
                List<OHLC> ohlcList = new();
                List<double> volumeList = new();

                if (jsonDoc.RootElement.TryGetProperty("last_indicators", out var indicatorsElement) && indicatorsElement.ValueKind == JsonValueKind.Array)
                {
                    var allIndicators = indicatorsElement.EnumerateArray().Reverse().ToList();
                    foreach (var indicator in allIndicators)
                    {
                        if (indicator.TryGetProperty("Open", out var o) && indicator.TryGetProperty("High", out var h) &&
                            indicator.TryGetProperty("Low", out var l) && indicator.TryGetProperty("Close", out var c) &&
                            indicator.TryGetProperty("Date", out var d) && DateTime.TryParse(d.GetString(), out DateTime dt))
                        {
                            ohlcList.Add(new OHLC(o.GetDouble(), h.GetDouble(), l.GetDouble(), c.GetDouble(), dt, TimeSpan.FromMinutes(15)));

                            double vol = 0;
                            if (indicator.TryGetProperty("Volume", out var v)) vol = v.GetDouble();
                            volumeList.Add(vol);
                        }
                    }
                }

                if (ohlcList.Count > 0)
                {
                    // --- 1. ÜST GRAFİK: MUMLAR VE MA ---
                    var candles = WpfPlotPrice.Plot.Add.Candlestick(ohlcList);
                    candles.RisingColor = ScottPlot.Color.FromHex("#24E500");
                    candles.FallingColor = ScottPlot.Color.FromHex("#ff0000");
                    candles.Axes.YAxis = WpfPlotPrice.Plot.Axes.Right;

                    // MA Hesapla ve Çiz
                    double[] xs = ohlcList.Select(x => x.DateTime.ToOADate()).ToArray();

                    var ma7Data = FinanceIndicators.CalculateSMA(ohlcList, 7);
                    var ma7Line = WpfPlotPrice.Plot.Add.ScatterLine(xs, ma7Data);
                    ma7Line.Color = ScottPlot.Color.FromHex("#F0B90B");
                    ma7Line.LineWidth = 1.5f;
                    ma7Line.Axes.YAxis = WpfPlotPrice.Plot.Axes.Right;

                    var ma25Data = FinanceIndicators.CalculateSMA(ohlcList, 25);
                    var ma25Line = WpfPlotPrice.Plot.Add.ScatterLine(xs, ma25Data);
                    ma25Line.Color = ScottPlot.Color.FromHex("#E0294A");
                    ma25Line.LineWidth = 1.5f;
                    ma25Line.Axes.YAxis = WpfPlotPrice.Plot.Axes.Right;

                    var ma99Data = FinanceIndicators.CalculateSMA(ohlcList, 99);
                    var ma99Line = WpfPlotPrice.Plot.Add.ScatterLine(xs, ma99Data);
                    ma99Line.Color = ScottPlot.Color.FromHex("#8F519B");
                    ma99Line.LineWidth = 1.5f;
                    ma99Line.Axes.YAxis = WpfPlotPrice.Plot.Axes.Right;

                    // --- 2. ALT GRAFİK: HACİM BARLARI ---
                    var bars = new List<ScottPlot.Bar>();
                    for (int i = 0; i < ohlcList.Count; i++)
                    {
                        var color = ohlcList[i].Close >= ohlcList[i].Open
                            ? ScottPlot.Color.FromHex("#0ECB8180")
                            : ScottPlot.Color.FromHex("#F6465D80");

                        var bar = new ScottPlot.Bar()
                        {
                            Position = ohlcList[i].DateTime.ToOADate(),
                            Value = volumeList[i],
                            ValueBase = 0,
                            FillColor = color,
                            Size = 0.007
                        };
                        bars.Add(bar);
                    }
                    var barPlot = WpfPlotVolume.Plot.Add.Bars(bars);
                    barPlot.Axes.YAxis = WpfPlotVolume.Plot.Axes.Right; // Hacim skalası sağda

                    // --- 3. EKSEN KİLİTLEME (SENKRONİZASYON) ---
                    // Otomatik ölçekle
                    WpfPlotPrice.Plot.Axes.AutoScale();
                    WpfPlotVolume.Plot.Axes.AutoScale();

                    // Alt grafiğin üst grafiği takip etmesini sağla (Zoom/Pan senkronizasyonu)
                    // ScottPlot 5'te eksenleri birbirine bağlama kuralı ekliyoruz
                    WpfPlotPrice.Plot.Axes.Link(WpfPlotVolume);
                    // VEYA tam tersi gerekebilir, test edelim. Genelde LinkX yeterlidir.
                    // Bu sayede üsttekini kaydırınca alttaki de kayar.

                    // Bilgi Şeridini Güncelle
                    var last = ohlcList.Last();
                    UpdateChartInfo(last, ma7Data.Last(), ma25Data.Last(), ma99Data.Last());

                    WpfPlotPrice.Refresh();
                    WpfPlotVolume.Refresh();
                    AppendToTerminal($"Grafik {symbol} için hazırlandı.");
                }
            }
            catch (Exception ex)
            {
                AppendToTerminal($"GRAFİK HATASI: {ex.Message}");
            }
        }

        private void UpdateChartInfo(OHLC candle, double ma7, double ma25, double ma99)
        {
            Dispatcher.Invoke(() =>
            {
                if (ChartHeaderInfo != null)
                {
                    ChartHeaderInfo.Text = $"O: {candle.Open:0.00} H: {candle.High:0.00} L: {candle.Low:0.00} C: {candle.Close:0.00}";
                }
                if (ChartIndicatorsInfo != null)
                {
                    ChartIndicatorsInfo.Text = $"MA(7): {ma7:0.00} MA(25): {ma25:0.00} MA(99): {ma99:0.00}";
                }
            });
        }

        private void StartPythonBackend()
        {
            try
            {
                string exePath = AppDomain.CurrentDomain.BaseDirectory;
                string projectPath = System.IO.Path.Combine(exePath, "CryptoBackend");

                if (!System.IO.Directory.Exists(projectPath))
                {
                    projectPath = System.IO.Path.GetFullPath(System.IO.Path.Combine(exePath, @"..\..\..\CryptoBackend"));
                }

                if (!System.IO.Directory.Exists(projectPath))
                {
                    AppendToTerminal("HATA: Python klasörü bulunamadı!");
                    System.Windows.MessageBox.Show("Python klasörü bulunamadı!", "Hata", MessageBoxButton.OK, MessageBoxImage.Error);
                    return;
                }

                var psi = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "cmd.exe",
                    Arguments = "/c python -m uvicorn api_services:app --reload",
                    WorkingDirectory = projectPath,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden,
                    StandardOutputEncoding = Encoding.UTF8,
                    StandardErrorEncoding = Encoding.UTF8
                };

                _pythonProcess = new System.Diagnostics.Process();
                _pythonProcess.StartInfo = psi;

                _pythonProcess.OutputDataReceived += (s, e) => AppendToTerminal(e.Data);
                _pythonProcess.ErrorDataReceived += (s, e) => AppendToTerminal(e.Data);

                _pythonProcess.Start();
                _pythonProcess.BeginOutputReadLine();
                _pythonProcess.BeginErrorReadLine();

                AppendToTerminal("SİSTEM BAŞLATILIYOR...");
            }
            catch (Exception ex)
            {
                AppendToTerminal($"KRİTİK HATA: {ex.Message}");
            }
        }

        private void AppendToTerminal(string? text)
        {
            if (string.IsNullOrEmpty(text)) return;
            Dispatcher.Invoke(() =>
            {
                if (TerminalOutput is not null)
                {
                    TerminalOutput.AppendText($"> {text}\n");
                    TerminalOutput.ScrollToEnd();
                }
            });
        }

        private void MainWindow_Closing(object? sender, System.ComponentModel.CancelEventArgs e) => KillPythonProcess();

        private void KillPythonProcess()
        {
            try
            {
                if (_pythonProcess is not null && !_pythonProcess.HasExited)
                {
                    System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = "taskkill",
                        Arguments = $"/F /T /PID {_pythonProcess.Id}",
                        CreateNoWindow = true,
                        UseShellExecute = false
                    });
                    _pythonProcess.Dispose();
                }
            }
            catch { }
        }

        private async void RefreshListButton_Click(object sender, RoutedEventArgs e)
        {
            if (RefreshListButton == null) return;

            RefreshListButton.IsEnabled = false;
            AppendToTerminal("------------------------------------------------");
            AppendToTerminal("↻ Liste verileri yerel veritabanından güncelleniyor...");

            try
            {
                // Sadece veriyi çek, tarama yapma
                await LoadMarketDataAsync(retryUntilSuccess: false);
                AppendToTerminal("✅ Liste başarıyla güncellendi.");
            }
            catch (Exception ex)
            {
                AppendToTerminal($"❌ LİSTE HATASI: {ex.Message}");
            }
            finally
            {
                RefreshListButton.IsEnabled = true;
            }
        }

        // --- YENİ: TARAMAYI BAŞLAT (YAVAŞ - API TETİKLER) ---
        private async void StartScanButton_Click(object sender, RoutedEventArgs e)
        {
            if (StartScanButton == null) return;

            // 1. UI HAZIRLIĞI: Butonu kilitle, Barı aç, Yazıyı değiştir
            StartScanButton.IsEnabled = false;
            ScanProgressBar.Visibility = Visibility.Visible; // Barı göster
            ScanButtonText.Text = "TARANIYOR...";            // Yazıyı güncelle

            AppendToTerminal("------------------------------------------------");
            AppendToTerminal("🚀 PİYASA TARAMASI BAŞLATILIYOR (BINANCE)...");
            AppendToTerminal("⏳ Bu işlem mum verilerini analiz ettiği için 3-5 dakika sürebilir. Lütfen bekleyin.");
            AppendToTerminal("------------------------------------------------");

            try
            {
                // 2. Python Tarama Endpoint'ini Tetikle (Uzun süren işlem)
                using (var scanClient = new HttpClient())
                {
                    scanClient.Timeout = TimeSpan.FromMinutes(20); // Uzun timeout
                    await scanClient.GetAsync(ScanUrl);
                }

                AppendToTerminal("✅ Tarama tamamlandı. Sonuçlar listeye çekiliyor...");

                // 3. Tarama bitince listeyi otomatik güncelle
                await LoadMarketDataAsync(retryUntilSuccess: false);

                System.Windows.MessageBox.Show("Piyasa Taraması ve Analizi Tamamlandı!", "Başarılı", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (Exception ex)
            {
                AppendToTerminal($"❌ TARAMA HATASI: {ex.Message}");
                System.Windows.MessageBox.Show("Tarama sırasında hata oluştu: " + ex.Message);
            }
            finally
            {
                // 4. TEMİZLİK: Barı gizle, Yazıyı eski haline getir, Butonu aç
                if (ScanProgressBar != null) ScanProgressBar.Visibility = Visibility.Collapsed;
                if (ScanButtonText != null) ScanButtonText.Text = "TARAMAYI BAŞLAT 🚀";

                StartScanButton.IsEnabled = true;
            }
        }

        private async Task LoadMarketDataAsync(bool retryUntilSuccess)
        {
            bool success = false;
            int retryCount = 0;

            do
            {
                try
                {
                    var response = await _httpClient.GetStringAsync(ApiUrl);

                    using (JsonDocument doc = JsonDocument.Parse(response))
                    {
                        var root = doc.RootElement;
                        if (root.TryGetProperty("data", out var dataElement))
                        {
                            var coinList = new List<CoinItem>();

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

                                coinList.Add(new()
                                {
                                    Symbol = symbol,
                                    Price = price,
                                    Volume = volume,
                                    AiPercentage = aiRate,
                                    ExpectedPrice = expectedPrice,
                                    SignalType = signalType,
                                    SignalColor = signalColor,
                                    RiskRatio = riskRatioText,
                                    RiskColor = riskColor,
                                    IsFavorite = false
                                });
                            }

                            _allCoinsBackup = coinList.ToList();
                            UpdateGrids(_allCoinsBackup);

                            success = true;
                            AppendToTerminal("Veri seti güncellendi.");
                        }
                    }
                }
                catch (Exception)
                {
                    if (retryUntilSuccess)
                    {
                        await Task.Delay(2000);
                        retryCount++;
                        if (retryCount > 30) break;
                    }
                    else
                    {
                        throw;
                    }
                }
            } while (retryUntilSuccess && !success);
        }

        private void SearchButton_Click(object sender, RoutedEventArgs e)
        {
            string searchText = SearchText.Text.Trim();

            if (_allCoinsBackup is null || _allCoinsBackup.Count == 0) return;

            List<CoinItem> filteredList;

            if (string.IsNullOrEmpty(searchText))
            {
                filteredList = _allCoinsBackup.ToList();
            }
            else
            {
                filteredList = _allCoinsBackup
                    .Where(x => x.Symbol.StartsWith(searchText, StringComparison.OrdinalIgnoreCase))
                    .ToList();
            }

            UpdateGrids(filteredList);
            MyDataGrid.Focus();
        }

        private void UpdateGrids(List<CoinItem> sourceList)
        {
            MyDataGrid.ItemsSource = sourceList.OrderByDescending(x => Math.Abs(x.AiPercentage)).ToList();
            MyDataGrid1.ItemsSource = sourceList.OrderByDescending(x => x.AiPercentage).ToList();
        }

        private void SetupDataGrids()
        {
            var grids = new[] { MyDataGrid, MyDataGrid1 };

            for (int i = 0; i < grids.Length; i++)
            {
                var grid = grids[i];
                grid.IsReadOnly = true;
                grid.AutoGenerateColumns = false;
                grid.Columns.Clear();
                grid.GridLinesVisibility = DataGridGridLinesVisibility.None;
                grid.RowHeight = 35;

                var leftStyle = new Style(typeof(TextBlock));
                leftStyle.Setters.Add(new Setter(TextBlock.TextAlignmentProperty, TextAlignment.Left));
                leftStyle.Setters.Add(new Setter(TextBlock.VerticalAlignmentProperty, System.Windows.VerticalAlignment.Center));
                leftStyle.Setters.Add(new Setter(TextBlock.PaddingProperty, new Thickness(10, 0, 0, 0)));

                grid.Columns.Add(new DataGridTextColumn { Header = "Coin", Binding = new Binding("Symbol"), Width = new DataGridLength(1, DataGridLengthUnitType.Star), FontWeight = FontWeights.Bold, ElementStyle = leftStyle });
                grid.Columns.Add(new DataGridTextColumn { Header = "Fiyat ($)", Binding = new Binding("Price"), Width = new DataGridLength(1, DataGridLengthUnitType.Auto), ElementStyle = leftStyle });

                var signalColumn = new DataGridTextColumn { Header = "İşlem", Binding = new Binding("SignalType"), Width = new DataGridLength(0.7, DataGridLengthUnitType.Star), FontWeight = FontWeights.Bold, ElementStyle = leftStyle };
                Style cellStyle = new Style(typeof(DataGridCell));
                cellStyle.Setters.Add(new Setter(DataGridCell.ForegroundProperty, new Binding("SignalColor")));

                var trigger = new Trigger { Property = DataGridCell.IsSelectedProperty, Value = true };
                trigger.Setters.Add(new Setter(DataGridCell.BackgroundProperty, (SolidColorBrush)new BrushConverter().ConvertFrom("#446A6A")));
                trigger.Setters.Add(new Setter(DataGridCell.BorderThicknessProperty, new Thickness(0)));
                cellStyle.Triggers.Add(trigger);

                signalColumn.CellStyle = cellStyle;
                grid.Columns.Add(signalColumn);

                if (i == 0)
                {
                    var riskColumn = new DataGridTextColumn { Header = "Risk", Binding = new Binding("RiskRatio"), Width = new DataGridLength(0.6, DataGridLengthUnitType.Star), FontWeight = FontWeights.Bold, ElementStyle = leftStyle };
                    Style riskStyle = new Style(typeof(DataGridCell));
                    riskStyle.Setters.Add(new Setter(DataGridCell.ForegroundProperty, new Binding("RiskColor")));

                    var riskTrigger = new Trigger { Property = DataGridCell.IsSelectedProperty, Value = true };
                    riskTrigger.Setters.Add(new Setter(DataGridCell.BackgroundProperty, (SolidColorBrush)new BrushConverter().ConvertFrom("#446A6A")));
                    riskTrigger.Setters.Add(new Setter(DataGridCell.BorderThicknessProperty, new Thickness(0)));
                    riskStyle.Triggers.Add(riskTrigger);

                    riskColumn.CellStyle = riskStyle;
                    grid.Columns.Add(riskColumn);
                }

                grid.Columns.Add(new DataGridTextColumn { Header = "Hedef ($)", Binding = new Binding("ExpectedPrice"), Width = new DataGridLength(1, DataGridLengthUnitType.Auto), ElementStyle = leftStyle });
                grid.Columns.Add(new DataGridTextColumn { Header = "AI Tahmini", Binding = new Binding("AiPercentage") { StringFormat = "P4" }, Width = new DataGridLength(1, DataGridLengthUnitType.Star), ElementStyle = leftStyle });
                grid.Columns.Add(new DataGridTextColumn { Header = "Hacim", Binding = new Binding("Volume") { StringFormat = "N0" }, Width = new DataGridLength(1, DataGridLengthUnitType.Star), ElementStyle = leftStyle });

                var starColumn = new DataGridTemplateColumn { Header = "Fav", Width = 40 };
                var factory = new FrameworkElementFactory(typeof(StarControl));
                factory.SetBinding(StarControl.IsStarredProperty, new Binding("IsFavorite"));
                starColumn.CellTemplate = new DataTemplate { VisualTree = factory };
                grid.Columns.Add(starColumn);
            }
        }

        private void BtnMinimize_Click(object sender, RoutedEventArgs e) { this.WindowState = WindowState.Minimized; }
        private void ExitFullScreen_Click(object sender, RoutedEventArgs e) { this.Close(); }

        private static void AnimateGridLength(ColumnDefinition column, double from, double to, double durationSeconds)
        {
            GridLengthAnimation anim = new() { From = new GridLength(from, GridUnitType.Star), To = new GridLength(to, GridUnitType.Star), Duration = TimeSpan.FromSeconds(durationSeconds) };
            column.BeginAnimation(ColumnDefinition.WidthProperty, anim);
        }

        public static T? FindVisualChild<T>(DependencyObject parent) where T : DependencyObject
        {
            if (parent is null) return null;
            for (int i = 0; i < VisualTreeHelper.GetChildrenCount(parent); i++)
            {
                if (VisualTreeHelper.GetChild(parent, i) is not { } child) continue;
                if (child is T tChild) return tChild;
                if (FindVisualChild<T>(child) is { } childOfChild) return childOfChild;
            }
            return null;
        }

        private void Border_MouseEnter1(object sender, MouseEventArgs e)
        {
            if (sender is Border border) { border.Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#446A6A")); if (FindVisualChild<StackPanel>(border) is { } stackPanel) { if (FindVisualChild<TextBlock>(stackPanel) is { } textBlock) textBlock.Foreground = new SolidColorBrush(Color.FromArgb(255, 100, 255, 255)); } }
        }
        private void Border_MouseLeave1(object sender, MouseEventArgs e)
        {
            if (sender is Border border) { border.Background = Brushes.Transparent; if (FindVisualChild<StackPanel>(border) is { } stackPanel) { if (FindVisualChild<TextBlock>(stackPanel) is { } textBlock) textBlock.Foreground = new SolidColorBrush(Color.FromArgb(255, 20, 71, 72)); } }
        }
        private void Border_MouseEnter(object sender, MouseEventArgs e)
        {
            AnimateGridLength(Solpanel, Solpanel.Width.Value, 2, 0.2);
            AnimateGridLength(Sagpanel, Sagpanel.Width.Value, 8, 0.2);
            var fadeIn = new DoubleAnimation(1, TimeSpan.FromSeconds(0.2));
            FavoriteText.BeginAnimation(UIElement.OpacityProperty, fadeIn);
            HomePAGEText.BeginAnimation(UIElement.OpacityProperty, fadeIn);
        }
        private void Border_MouseLeave(object sender, MouseEventArgs e)
        {
            AnimateGridLength(Solpanel, Solpanel.Width.Value, 0.8, 0.5);
            AnimateGridLength(Sagpanel, Sagpanel.Width.Value, 9.2, 0.5);
            var fadeOut = new DoubleAnimation(0, TimeSpan.FromSeconds(0.5));
            FavoriteText.BeginAnimation(UIElement.OpacityProperty, fadeOut);
            HomePAGEText.BeginAnimation(UIElement.OpacityProperty, fadeOut);
        }

        private Image? activeImage = null;
        private void Border_MouseLeftButtonDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            var clickedBorder = sender as Border;
            if (clickedBorder is null) return;
            Image? clickedImage = null;
            if (clickedBorder.Name == "Home") clickedImage = HomeImage;
            else if (clickedBorder.Name == "Favori") clickedImage = StarImage;
            if (clickedImage is null || activeImage == clickedImage) return;
            if (activeImage is not null) FadeChangeImage(activeImage, GetNormalSource(activeImage));
            FadeChangeImage(clickedImage, GetActiveSource(clickedImage));
            activeImage = clickedImage;
        }
        private static void FadeChangeImage(Image image, string? newSource)
        {
            if (newSource is null) return;
            var fadeOut = new DoubleAnimation(1, 0, TimeSpan.FromMilliseconds(150));
            fadeOut.Completed += (s, a) => { image.Source = new BitmapImage(new Uri(newSource, UriKind.Relative)); var fadeIn = new DoubleAnimation(0, 1, TimeSpan.FromMilliseconds(150)); image.BeginAnimation(UIElement.OpacityProperty, fadeIn); };
            image.BeginAnimation(UIElement.OpacityProperty, fadeOut);
        }
        private string? GetActiveSource(Image image) { return image == StarImage ? "/IMGS/LStar.png" : image == HomeImage ? "/IMGS/Lhomepage.png" : null; }
        private string? GetNormalSource(Image image) { return image == StarImage ? "/IMGS/Star.png" : image == HomeImage ? "/IMGS/homepage.png" : null; }
    }

    public class CoinItem
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
        public bool IsFavorite { get; set; } = false;
    }
}