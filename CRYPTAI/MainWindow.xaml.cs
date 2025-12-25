using CRYPTAI.Helpers;
using CRYPTAI.Models;
using CRYPTAI.Services;
using ScottPlot;
using System.Net.Http;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;
using Application = System.Windows.Application;
using Binding = System.Windows.Data.Binding;
using Brushes = System.Windows.Media.Brushes;
using Color = System.Windows.Media.Color;
using ColorConverter = System.Windows.Media.ColorConverter;
using Colors = System.Windows.Media.Colors;
using MouseEventArgs = System.Windows.Input.MouseEventArgs;
using System.Linq;

namespace CRYPTAI
{
    public partial class MainWindow : Window
    {
        // --- SERVİSLER ---
        private readonly PythonBackendService _backendService;
        private readonly MarketDataService _marketService;

        // --- UI DEĞİŞKENLERİ ---
        private List<CoinItem> _allCoinsBackup = new();
        private readonly HttpClient _chartHttpClient = new();

        private double _lastLongPercentage = 50.0;

        public MainWindow()
        {
            InitializeComponent();

            _marketService = new MarketDataService();
            _backendService = new PythonBackendService();

            _backendService.OnLogReceived += AppendToTerminal;
            _backendService.OnRecoveryNeeded += ShowRepairDialog;

            _backendService.OnFatalError += (errorMsg) =>
            {
                Dispatcher.Invoke(() =>
                {
                    if (LoadingGrid != null) LoadingGrid.Visibility = Visibility.Collapsed;
                    this.IsEnabled = true;
                    ShowAlertDialog("KRİTİK HATA", errorMsg, isError: true);
                });
            };

            SetupDataGrids();

            if (HomePAGEText != null) HomePAGEText.Opacity = 0;
            Task.Run(() => _backendService.StartBackend());
            this.Closing += MainWindow_Closing;
            Application.Current.Exit += (s, e) => _backendService.KillProcess();
            ShowStartupBanner();
        }

        // --- YARDIMCI METOTLAR ---
        private void ShowAlertDialog(string title, string message, bool isError = false)
        {
            Dispatcher.Invoke(() =>
            {
                var alert = new AlertDialog(title, message, isError);
                alert.Owner = this;
                alert.ShowDialog();
            });
        }

        private void ShowRepairDialog()
        {
            Dispatcher.Invoke(() =>
            {
                _backendService.KillProcess();
                RepairDialog dialog = new RepairDialog();
                dialog.Owner = this;
                bool? result = dialog.ShowDialog();

                if (dialog.IsRepairRequested)
                {
                    if (LoadingGrid != null) LoadingGrid.Visibility = Visibility.Visible;
                    this.IsEnabled = false;

                    Properties.Settings.Default.IsDependenciesInstalled = false;
                    Properties.Settings.Default.Save();

                    AppendToTerminal("------------------------------------------------");
                    AppendToTerminal("🛠️ SİSTEM ONARIMI BAŞLATILDI");
                    AppendToTerminal("📦 Gerekli kütüphaneler indiriliyor...");

                    Task.Run(async () =>
                    {
                        try
                        {
                            await _backendService.StartBackend(forceCheck: true);
                            AppendToTerminal("✅ KURULUM BAŞARIYLA TAMAMLANDI.");

                            await Task.Delay(2000);
                            Dispatcher.Invoke(() =>
                            {
                                string appPath = Environment.ProcessPath ?? System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName;
                                System.Diagnostics.Process.Start(appPath);
                                Application.Current.Shutdown();
                            });
                        }
                        catch (Exception ex)
                        {
                            Dispatcher.Invoke(() =>
                            {
                                if (LoadingGrid != null) LoadingGrid.Visibility = Visibility.Collapsed;
                                this.IsEnabled = true;
                                ShowAlertDialog("KRİTİK HATA", ex.Message, true);
                            });
                        }
                    });
                }
                else
                {
                    Application.Current.Shutdown();
                }
            });
        }

        private async void MainWin_Loaded(object sender, RoutedEventArgs e)
        {
            Sagpanel.Width = new GridLength(9.2, GridUnitType.Star);
            Solpanel.Width = new GridLength(0.8, GridUnitType.Star);

            UpdateSentiment(50);

            if (LoadingGrid is not null) LoadingGrid.Visibility = Visibility.Visible;
            await RefreshMarketData(retryUntilSuccess: true);
            if (LoadingGrid is not null) LoadingGrid.Visibility = Visibility.Collapsed;
        }

        private async Task RefreshMarketData(bool retryUntilSuccess)
        {
            bool success = false;
            int retryCount = 0;

            do
            {
                try
                {
                    var data = await _marketService.GetMarketDataAsync();
                    _allCoinsBackup = data;
                    ApplyFilters();

                    CalculateGlobalSentiment();

                    success = true;
                }
                catch (Exception ex)
                {
                    if (retryUntilSuccess)
                    {
                        await Task.Delay(2000);
                        retryCount++;
                        if (retryCount > 30) break;
                    }
                    else
                    {
                        AppendToTerminal($"VERİ HATASI: {ex.Message}");
                        break;
                    }
                }
            } while (retryUntilSuccess && !success);
        }

        private void ApplyFilters()
        {
            if (_allCoinsBackup is null || _allCoinsBackup.Count == 0) return;

            string searchText = SearchText.Text.Trim();
            var query = _allCoinsBackup.AsEnumerable();

            if (!string.IsNullOrEmpty(searchText))
            {
                query = query.Where(x => x.Symbol.StartsWith(searchText, StringComparison.OrdinalIgnoreCase));
            }
            UpdateGrids(query.ToList());
        }

        private void UpdateGrids(List<CoinItem> sourceList)
        {
            MyDataGrid.ItemsSource = sourceList.OrderByDescending(x => Math.Abs(x.AiPercentage)).ToList();
        }

        private void CalculateGlobalSentiment()
        {
            if (_allCoinsBackup == null || _allCoinsBackup.Count == 0) return;
            var activeCoins = _allCoinsBackup.Where(c => c.AiPercentage != 0).ToList();
            double totalCount = activeCoins.Count;

            if (totalCount == 0)
            {
                UpdateSentiment(50);
                return;
            }
            double longCount = activeCoins.Count(c => c.AiPercentage > 0);
            double longRatio = (longCount / totalCount) * 100.0;

            UpdateSentiment(longRatio);
        }

        public void UpdateSentiment(double longPercentage)
        {
            if (longPercentage > 100) longPercentage = 100;
            if (longPercentage < 0) longPercentage = 0;

            _lastLongPercentage = longPercentage; 
            double shortPercentage = 100.0 - longPercentage;

            Dispatcher.Invoke(() =>
            {
                if (ColBull != null) ColBull.Width = new GridLength(longPercentage, GridUnitType.Star);
                if (ColBear != null) ColBear.Width = new GridLength(shortPercentage, GridUnitType.Star);
                if (TxtLongRate != null) TxtLongRate.Text = $"%{longPercentage:0.0}";
                if (TxtShortRate != null) TxtShortRate.Text = $"%{shortPercentage:0.0}";
                if (TxtLongRate?.Parent is StackPanel pLong) pLong.Opacity = longPercentage < 15 ? 0 : 1;
                if (TxtShortRate?.Parent is StackPanel pShort) pShort.Opacity = shortPercentage < 15 ? 0 : 1;
                if (BorderLong != null)
                    BorderLong.CornerRadius = (longPercentage >= 99.9) ? new CornerRadius(4) : new CornerRadius(4, 0, 0, 4);

                if (BorderShort != null)
                    BorderShort.CornerRadius = (shortPercentage >= 99.9) ? new CornerRadius(4) : new CornerRadius(0, 4, 4, 0);
                UpdateIconPosition();
            });
        }

        private void UpdateIconPosition()
        {
            if (SentimentBorder == null || IconContainer == null) return;

            double totalWidth = SentimentBorder.ActualWidth;
            if (totalWidth <= 0) return;

            double containerWidth = IconContainer.Width;
            if (double.IsNaN(containerWidth)) containerWidth = 30;

            double targetCenter = totalWidth * (_lastLongPercentage / 100.0);

            double leftPosition = targetCenter - (containerWidth / 2);

            Canvas.SetLeft(IconContainer, leftPosition);
        }
        private void SentimentBorder_SizeChanged(object sender, SizeChangedEventArgs e) { UpdateIconPosition();  }

        private async void RefreshListButton_Click(object sender, RoutedEventArgs e)
        {
            if (RefreshListButton == null) return;
            RefreshListButton.IsEnabled = false;
            AppendToTerminal("------------------------------------------------");
            AppendToTerminal("↻ Liste verileri yerel veritabanından güncelleniyor...");
            await RefreshMarketData(retryUntilSuccess: false);
            RefreshListButton.IsEnabled = true;
            ShowAlertDialog("Liste Yenilendi", "Veri seti güncellendi");
        }

        private async void StartScanButton_Click(object sender, RoutedEventArgs e)
        {
            if (StartScanButton == null) return;

            string selectedExchange = "binance";
            if (borsaCombobox.SelectedItem is ComboBoxItem borsaItem && borsaItem.Content != null)
                selectedExchange = borsaItem.Content.ToString()!.ToLowerInvariant();

            string selectedTimeframe = "15m";
            if (timeFrameCombo.SelectedItem is ComboBoxItem timeItem && timeItem.Content != null)
                selectedTimeframe = timeItem.Content.ToString()!.ToLowerInvariant();

            StartScanButton.IsEnabled = false;
            ScanProgressBar.Visibility = Visibility.Visible;
            ScanButtonText.Text = $"{selectedExchange.ToUpper()} TARANIYOR...";

            AppendToTerminal("------------------------------------------------");
            AppendToTerminal($"🚀 PİYASA TARAMASI BAŞLATILIYOR ({selectedExchange.ToUpper()} - {selectedTimeframe})...");
            AppendToTerminal("⏳ Bu işlem 3-5 dakika sürebilir. Bekleyin.");

            try
            {
                await _marketService.TriggerScanAsync(selectedTimeframe, selectedExchange);
                AppendToTerminal("✅ Tarama tamamlandı. Sonuçlar listeye çekiliyor...");
                await RefreshMarketData(retryUntilSuccess: false);
                ShowAlertDialog("TARAMA TAMAMLANDI", "Veriler güncellendi.");
            }
            catch (Exception ex)
            {
                string errorMsg = ex.Message;
                if (errorMsg.Contains("500"))
                {
                    errorMsg += "\n⚠️ İPUCU: Python penceresini kapatıp programı yeniden başlatmayı deneyin.";
                }
                AppendToTerminal($"❌ TARAMA HATASI: {errorMsg}");
                ShowAlertDialog("HATA", errorMsg, isError: true);
            }
            finally
            {
                if (ScanProgressBar != null) ScanProgressBar.Visibility = Visibility.Collapsed;
                if (ScanButtonText != null) ScanButtonText.Text = "TARAMAYI BAŞLAT 🚀";
                StartScanButton.IsEnabled = true;
            }
        }

        private void SearchButton_Click(object sender, RoutedEventArgs e) => ApplyFilters();

        private async void MyDataGrid_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (MyDataGrid.SelectedItem is CoinItem selectedCoin)
            {
                await LoadAndShowCoinChart(selectedCoin.Symbol);
            }
        }

        private void BackToListViewButton_Click(object sender, RoutedEventArgs e)
        {
            if (ListAndTerminalContainer != null) ListAndTerminalContainer.Visibility = Visibility.Visible;
            if (ChartContainer != null) ChartContainer.Visibility = Visibility.Collapsed;
        }

        private void MainWindow_Closing(object? sender, System.ComponentModel.CancelEventArgs e) => _backendService.KillProcess();

        private void AppendToTerminal(string? text)
        {
            if (string.IsNullOrEmpty(text)) return;
            Dispatcher.Invoke(() =>
            {
                if (TerminalOutput != null)
                {
                    TerminalOutput.AppendText($"> {text}\n");
                    TerminalOutput.ScrollToEnd();
                }
            });
        }

        // --- GRAFİK KODLARI ---
        private async Task LoadAndShowCoinChart(string symbol)
        {
            if (ListAndTerminalContainer != null) ListAndTerminalContainer.Visibility = Visibility.Collapsed;
            if (ChartContainer != null) ChartContainer.Visibility = Visibility.Visible;

            ScottPlot.Color bgColor = ScottPlot.Color.FromHex("#161A25");
            ScottPlot.Color gridColor = ScottPlot.Color.FromHex("#2B2F36");
            ScottPlot.Color axisColor = ScottPlot.Color.FromHex("#848E9C");

            WpfPlotPrice.Plot.Clear();
            WpfPlotVolume.Plot.Clear();

            // Fiyat Grafiği
            WpfPlotPrice.Plot.FigureBackground.Color = bgColor;
            WpfPlotPrice.Plot.DataBackground.Color = bgColor;
            WpfPlotPrice.Plot.Grid.MajorLineColor = gridColor;
            WpfPlotPrice.Plot.Axes.Color(axisColor);
            WpfPlotPrice.Plot.Axes.Left.IsVisible = false;
            WpfPlotPrice.Plot.Axes.Right.IsVisible = true;
            WpfPlotPrice.Plot.Axes.Right.TickLabelStyle.ForeColor = axisColor;
            WpfPlotPrice.Plot.Axes.Bottom.TickLabelStyle.IsVisible = false;

            // Hacim Grafiği
            WpfPlotVolume.Plot.FigureBackground.Color = bgColor;
            WpfPlotVolume.Plot.DataBackground.Color = bgColor;
            WpfPlotVolume.Plot.Grid.MajorLineColor = gridColor;
            WpfPlotVolume.Plot.Axes.Color(axisColor);
            WpfPlotVolume.Plot.Axes.Left.IsVisible = false;
            WpfPlotVolume.Plot.Axes.Right.IsVisible = true;
            WpfPlotVolume.Plot.Axes.Right.TickLabelStyle.ForeColor = axisColor;
            WpfPlotVolume.Plot.Axes.DateTimeTicksBottom();

            WpfPlotPrice.Plot.Title("");
            WpfPlotVolume.Plot.Title("");
            WpfPlotPrice.Refresh();
            WpfPlotVolume.Refresh();

            AppendToTerminal($"Grafik verisi çekiliyor: {symbol}...");
            string apiSymbol = symbol.Replace("/USDT", "");

            try
            {
                var response = await _chartHttpClient.GetStringAsync($"http://127.0.0.1:8000/get_coin_data/{apiSymbol}");
                using var jsonDoc = JsonDocument.Parse(response);

                if (jsonDoc.RootElement.ValueKind != JsonValueKind.Object) { WpfPlotPrice.Plot.Title("HATA: Veri Yok"); return; }

                List<OHLC> rawList = new();
                List<double> volumeList = new();

                if (jsonDoc.RootElement.TryGetProperty("last_indicators", out var indicatorsElement) && indicatorsElement.ValueKind == JsonValueKind.Array)
                {
                    foreach (var indicator in indicatorsElement.EnumerateArray().Reverse())
                    {
                        if (indicator.TryGetProperty("Open", out var o) && indicator.TryGetProperty("High", out var h) &&
                            indicator.TryGetProperty("Low", out var l) && indicator.TryGetProperty("Close", out var c) &&
                            indicator.TryGetProperty("Date", out var d) && DateTime.TryParse(d.GetString(), out DateTime dt))
                        {
                            rawList.Add(new OHLC(o.GetDouble(), h.GetDouble(), l.GetDouble(), c.GetDouble(), dt, TimeSpan.Zero));

                            double vol = 0;
                            if (indicator.TryGetProperty("Volume", out var v)) vol = v.GetDouble();
                            volumeList.Add(vol);
                        }
                    }
                }

                if (rawList.Count > 0)
                {
                    var sortedData = rawList.Zip(volumeList, (ohlc, vol) => new { ohlc, vol })
                                            .OrderBy(x => x.ohlc.DateTime)
                                            .ToList();

                    TimeSpan detectedInterval = TimeSpan.FromMinutes(15);
                    if (sortedData.Count > 1)
                    {
                        var diff = sortedData[1].ohlc.DateTime - sortedData[0].ohlc.DateTime;
                        if (diff.TotalMinutes > 0) detectedInterval = diff;
                    }

                    List<OHLC> finalOhlcList = new();
                    List<double> finalVolList = new();

                    foreach (var item in sortedData)
                    {
                        finalOhlcList.Add(new OHLC(item.ohlc.Open, item.ohlc.High, item.ohlc.Low, item.ohlc.Close, item.ohlc.DateTime, detectedInterval));
                        finalVolList.Add(item.vol);
                    }

                    var candles = WpfPlotPrice.Plot.Add.Candlestick(finalOhlcList);
                    candles.RisingColor = ScottPlot.Color.FromHex("#24E500");
                    candles.FallingColor = ScottPlot.Color.FromHex("#ff0000");
                    candles.Axes.YAxis = WpfPlotPrice.Plot.Axes.Right;

                    double[] xs = finalOhlcList.Select(x => x.DateTime.ToOADate()).ToArray();
                    if (finalOhlcList.Count >= 7) AddLineToChart(WpfPlotPrice, xs, FinanceIndicators.CalculateSMA(finalOhlcList, 7), "#F0B90B");
                    if (finalOhlcList.Count >= 25) AddLineToChart(WpfPlotPrice, xs, FinanceIndicators.CalculateSMA(finalOhlcList, 25), "#E0294A");
                    if (finalOhlcList.Count >= 99) AddLineToChart(WpfPlotPrice, xs, FinanceIndicators.CalculateSMA(finalOhlcList, 99), "#8F519B");

                    double barWidth = detectedInterval.TotalDays * 0.85;
                    var bars = new List<ScottPlot.Bar>();
                    for (int i = 0; i < finalOhlcList.Count; i++)
                    {
                        var color = finalOhlcList[i].Close >= finalOhlcList[i].Open ? ScottPlot.Color.FromHex("#0ECB8180") : ScottPlot.Color.FromHex("#F6465D80");
                        bars.Add(new ScottPlot.Bar { Position = finalOhlcList[i].DateTime.ToOADate(), Value = finalVolList[i], ValueBase = 0, FillColor = color, Size = barWidth });
                    }
                    var barPlot = WpfPlotVolume.Plot.Add.Bars(bars);
                    barPlot.Axes.YAxis = WpfPlotVolume.Plot.Axes.Right;

                    WpfPlotPrice.Plot.Axes.AutoScale();
                    WpfPlotVolume.Plot.Axes.AutoScale();

                    WpfPlotPrice.Plot.Axes.Link(WpfPlotVolume, x: true, y: false);

                    UpdateChartInfo(finalOhlcList.Last(), 0, 0, 0);
                    WpfPlotPrice.Refresh();
                    WpfPlotVolume.Refresh();
                    AppendToTerminal($"Grafik {symbol} için hazırlandı.");
                }
            }
            catch (Exception ex) { AppendToTerminal($"GRAFİK HATA: {ex.Message}"); }
        }

        private void AddLineToChart(IPlotControl plot, double[] xs, double[] ys, string hexColor)
        {
            var line = plot.Plot.Add.ScatterLine(xs, ys);
            line.Color = ScottPlot.Color.FromHex(hexColor);
            line.LineWidth = 1.5f;
            line.Axes.YAxis = plot.Plot.Axes.Right;
        }

        private void UpdateChartInfo(OHLC candle, double ma7, double ma25, double ma99)
        {
            Dispatcher.Invoke(() =>
            {
                if (ChartHeaderInfo != null) ChartHeaderInfo.Text = $"O: {candle.Open:0.00} H: {candle.High:0.00} L: {candle.Low:0.00} C: {candle.Close:0.00}";
                if (ChartIndicatorsInfo != null) ChartIndicatorsInfo.Text = $"MA(7): {ma7:0.00} MA(25): {ma25:0.00} MA(99): {ma99:0.00}";
            });
        }

        private void ShowStartupBanner()
        {
            string banner = @"
_________________________________________________

      .---.                  CRYPTAI  
     /   _ \         INTELLIGENT TRADING SYSTEM
 o--|  |_) |       v1.0.4 - Stable Build
    |   _ <        Powered by: Python AI & .NET 8
 o--|  |_) |      
     \_____/          SYSTEM STATUS: [ONLINE]
                      MODE: AUTOMATIC
_________________________________________________
";
            Dispatcher.Invoke(() =>
            {
                TerminalOutput.AppendText(banner + "\n");
                TerminalOutput.ScrollToEnd();
            });
        }

        private void SetupDataGrids()
        {
            var grids = new[] { MyDataGrid };
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

                var centerStyle = new Style(typeof(TextBlock));
                centerStyle.Setters.Add(new Setter(TextBlock.TextAlignmentProperty, TextAlignment.Center));
                centerStyle.Setters.Add(new Setter(TextBlock.VerticalAlignmentProperty, System.Windows.VerticalAlignment.Center));

                grid.Columns.Add(new DataGridTextColumn { Header = "Coin", Binding = new Binding("Symbol"), Width = new DataGridLength(1, DataGridLengthUnitType.Star), FontWeight = FontWeights.Bold, ElementStyle = leftStyle });
                grid.Columns.Add(new DataGridTextColumn { Header = "Fiyat ($)", Binding = new Binding("Price"), Width = new DataGridLength(1, DataGridLengthUnitType.Auto), ElementStyle = centerStyle });

                var signalColumn = new DataGridTextColumn { Header = "İşlem", Binding = new Binding("SignalType"), Width = new DataGridLength(0.7, DataGridLengthUnitType.Star), FontWeight = FontWeights.Bold, ElementStyle = centerStyle };
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
                    var riskColumn = new DataGridTextColumn { Header = "Risk", Binding = new Binding("RiskRatio"), Width = new DataGridLength(0.6, DataGridLengthUnitType.Star), FontWeight = FontWeights.Bold, ElementStyle = centerStyle };
                    Style riskStyle = new Style(typeof(DataGridCell));
                    riskStyle.Setters.Add(new Setter(DataGridCell.ForegroundProperty, new Binding("RiskColor")));
                    var riskTrigger = new Trigger { Property = DataGridCell.IsSelectedProperty, Value = true };
                    riskTrigger.Setters.Add(new Setter(DataGridCell.BackgroundProperty, (SolidColorBrush)new BrushConverter().ConvertFrom("#446A6A")));
                    riskTrigger.Setters.Add(new Setter(DataGridCell.BorderThicknessProperty, new Thickness(0)));
                    riskStyle.Triggers.Add(riskTrigger);
                    riskColumn.CellStyle = riskStyle;
                    grid.Columns.Add(riskColumn);
                }

                grid.Columns.Add(new DataGridTextColumn { Header = "Hedef ($)", Binding = new Binding("ExpectedPrice"), Width = new DataGridLength(1, DataGridLengthUnitType.Auto), ElementStyle = centerStyle });
                grid.Columns.Add(new DataGridTextColumn { Header = "AI Tahmini", Binding = new Binding("AiPercentage") { StringFormat = "P4" }, Width = new DataGridLength(1, DataGridLengthUnitType.Star), ElementStyle = centerStyle });
                grid.Columns.Add(new DataGridTextColumn { Header = "Hacim", Binding = new Binding("Volume") { StringFormat = "N0" }, Width = new DataGridLength(1, DataGridLengthUnitType.Star), ElementStyle = centerStyle });
            }
        }

        private void BtnMinimize_Click(object sender, RoutedEventArgs e) => this.WindowState = WindowState.Minimized;
        private void ExitFullScreen_Click(object sender, RoutedEventArgs e) => this.Close();

        private void Border_MouseEnter1(object sender, MouseEventArgs e) =>
            SetBorderColor(sender, "#446A6A", (Color)ColorConverter.ConvertFromString("#FF0B65B0"));
        private void Border_MouseLeave1(object sender, MouseEventArgs e) =>
            SetBorderColor(sender, "Transparent", Colors.White);

        private void SetBorderColor(object sender, string hexBg, Color textColor)
        {
            if (sender is Border border)
            {
                border.Background = hexBg == "Transparent" ? Brushes.Transparent : new SolidColorBrush((Color)ColorConverter.ConvertFromString(hexBg));
                if (FindVisualChild<StackPanel>(border) is { } stackPanel && FindVisualChild<TextBlock>(stackPanel) is { } textBlock)
                    textBlock.Foreground = new SolidColorBrush(textColor);
            }
        }

        private void Border_MouseEnter(object sender, MouseEventArgs e) => AnimateSidebar(2, 8, 1);
        private void Border_MouseLeave(object sender, MouseEventArgs e) => AnimateSidebar(0.8, 9.2, 0);

        private void AnimateSidebar(double left, double right, double opacity)
        {
            AnimateGridLength(Solpanel, Solpanel.Width.Value, left, 0.3);
            AnimateGridLength(Sagpanel, Sagpanel.Width.Value, right, 0.3);
            var anim = new DoubleAnimation(opacity, TimeSpan.FromSeconds(0.3));
            if (HomePAGEText != null) HomePAGEText.BeginAnimation(UIElement.OpacityProperty, anim);
        }

        private static void AnimateGridLength(ColumnDefinition column, double from, double to, double durationSeconds)
        {
            var anim = new GridLengthAnimation { From = new GridLength(from, GridUnitType.Star), To = new GridLength(to, GridUnitType.Star), Duration = TimeSpan.FromSeconds(durationSeconds) };
            column.BeginAnimation(ColumnDefinition.WidthProperty, anim);
        }

        public static T? FindVisualChild<T>(DependencyObject parent) where T : DependencyObject
        {
            if (parent is null) return null;
            for (int i = 0; i < VisualTreeHelper.GetChildrenCount(parent); i++)
            {
                if (VisualTreeHelper.GetChild(parent, i) is T tChild) return tChild;
                if (FindVisualChild<T>(VisualTreeHelper.GetChild(parent, i)) is { } childOfChild) return childOfChild;
            }
            return null;
        }

        private void SearchText_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter) ApplyFilters();
        }
    }
}