using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace CRYPTAI.Services
{
    public class PythonBackendService
    {
        private Process? _pythonProcess;

        // Logları arayüze göndermek için olay
        public event Action<string>? OnLogReceived;

        // KRİTİK: Otomatik onarım (pip install) gerektiğinde tetiklenir
        public event Action? OnRecoveryNeeded;

        // KRİTİK: Onarım işe yaramadıysa veya durum kötüyse tetiklenir (Alert Dialog için)
        public event Action<string>? OnFatalError;

        // Sonsuz döngüye girmemesi için bayraklar
        private bool _isRecoveryTriggered = false;
        private DateTime _lastErrorTime = DateTime.MinValue;

        public async Task StartBackend(bool forceCheck = false)
        {
            await Task.Run(() =>
            {
                try
                {
                    _isRecoveryTriggered = false;

                    string exePath = AppDomain.CurrentDomain.BaseDirectory;
                    string projectPath = Path.Combine(exePath, "CryptoBackend");

                    // Geliştirme ortamı vs. Canlı ortam yol kontrolü
                    if (!Directory.Exists(projectPath)) projectPath = Path.GetFullPath(Path.Combine(exePath, @"..\..\..\CryptoBackend"));

                    if (!Directory.Exists(projectPath))
                    {
                        string msg = "KRİTİK HATA: Backend dizini bulunamadı! 'CryptoBackend' klasörü eksik.";
                        OnLogReceived?.Invoke(msg);
                        OnFatalError?.Invoke(msg);
                        return;
                    }

                    // --- KÜTÜPHANE KONTROLÜ ---
                    if (forceCheck || !Properties.Settings.Default.IsDependenciesInstalled)
                    {
                        string reqFile = Path.Combine(projectPath, "requirements.txt");
                        if (File.Exists(reqFile))
                        {
                            LogToUi("------------------------------------------------");
                            LogToUi("SİSTEM: Eksik kütüphaneler tespit edildi. Kurulum başlıyor...");
                            LogToUi("PIP: İnternet hızınıza göre bu işlem 5-15 dakika sürebilir.");

                            bool installSuccess = InstallDependencies(projectPath);

                            if (installSuccess)
                            {
                                LogToUi("✅ SİSTEM: Tüm kütüphaneler başarıyla kuruldu.");
                                Properties.Settings.Default.IsDependenciesInstalled = true;
                                Properties.Settings.Default.Save();
                            }
                            else
                            {
                                LogToUi("⚠️ UYARI: Kurulum sırasında hatalar oluştu. Sistem yine de başlatılıyor...");
                            }
                            LogToUi("------------------------------------------------");
                        }
                    }
                    else
                    {
                        LogToUi("SİSTEM: Kütüphaneler hazır (Önbellek).");
                    }

                    // --- REVİZE EDİLDİ: GÜVENLİ PROCESS KONTROLÜ ---
                    // Eski process dispose edilmişse erişim hatası vermemesi için try-catch eklendi
                    bool isProcessRunning = false;
                    try
                    {
                        isProcessRunning = _pythonProcess != null && !_pythonProcess.HasExited;
                    }
                    catch
                    {
                        // Eğer Handle hatası alırsak process zaten ölüdür, değişkene null basar geçeriz.
                        _pythonProcess = null;
                        isProcessRunning = false;
                    }

                    if (isProcessRunning)
                    {
                        LogToUi("SİSTEM: Servis zaten çalışıyor.");
                        return;
                    }

                    // --- PYTHON SUNUCUSUNU BAŞLAT ---
                    var psi = new ProcessStartInfo
                    {
                        FileName = "cmd.exe",
                        Arguments = "/c python -m uvicorn api_services:app --reload",
                        WorkingDirectory = projectPath,
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true,
                        WindowStyle = ProcessWindowStyle.Hidden,
                        StandardOutputEncoding = Encoding.UTF8,
                        StandardErrorEncoding = Encoding.UTF8
                    };

                    psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
                    psi.EnvironmentVariables["PYTHONUTF8"] = "1";

                    _pythonProcess = new Process();
                    _pythonProcess.StartInfo = psi;

                    _pythonProcess.OutputDataReceived += (s, e) => HandleProcessOutput(e.Data, false);
                    _pythonProcess.ErrorDataReceived += (s, e) => HandleProcessOutput(e.Data, true);

                    _pythonProcess.Start();
                    _pythonProcess.BeginOutputReadLine();
                    _pythonProcess.BeginErrorReadLine();

                    LogToUi("BACKEND: Sunucu başlatılıyor... (AI Modülleri Yükleniyor)");
                }
                catch (Exception ex)
                {
                    string errorMsg = $"SYSTEM FAILURE: {ex.Message}";
                    LogToUi(errorMsg);
                    OnFatalError?.Invoke(errorMsg);
                }
            });
        }

        private void HandleProcessOutput(string? data, bool isErrorStream)
        {
            if (string.IsNullOrEmpty(data)) return;

            string prefix = isErrorStream ? "> " : "";
            OnLogReceived?.Invoke($"{prefix}{data}");

            AnalyzeErrorForRecovery(data);
        }

        private void AnalyzeErrorForRecovery(string logLine)
        {
            if (_isRecoveryTriggered) return;

            var match = Regex.Match(logLine, @"No module named ['""]([^'""]+)['""]", RegexOptions.IgnoreCase);
            bool isImportError = logLine.Contains("ImportError") || logLine.Contains("ModuleNotFoundError");

            if (match.Success || isImportError)
            {
                string missingModule = match.Success ? match.Groups[1].Value : "Bilinmeyen Modül";

                if ((DateTime.Now - _lastErrorTime).TotalSeconds < 10) return;
                _lastErrorTime = DateTime.Now;

                LogToUi($"❌ KRİTİK EKSİK TESPİT EDİLDİ: {missingModule}");

                // Her iki durumda da onarım tetikle
                _isRecoveryTriggered = true;
                OnRecoveryNeeded?.Invoke();
            }
        }

        // --- REVİZE EDİLDİ: PROCESS TEMİZLİĞİ ---
        public void KillProcess()
        {
            try
            {
                if (_pythonProcess is not null)
                {
                    try
                    {
                        if (!_pythonProcess.HasExited)
                        {
                            Process.Start(new ProcessStartInfo
                            {
                                FileName = "taskkill",
                                Arguments = $"/F /T /PID {_pythonProcess.Id}",
                                CreateNoWindow = true,
                                UseShellExecute = false
                            });
                        }
                    }
                    catch { /* Erişim reddedildi veya process zaten kapandı */ }

                    _pythonProcess.Dispose();
                }
            }
            catch { /* Genel hata yutucu */ }
            finally
            {
                // EN ÖNEMLİ KISIM: Değişkeni null yap ki StartBackend temiz başlasın
                _pythonProcess = null;
            }
        }

        private bool InstallDependencies(string workingDir)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "cmd.exe",
                    Arguments = "/c pip install -r requirements.txt --disable-pip-version-check --no-cache-dir",
                    WorkingDirectory = workingDir,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using (var process = new Process())
                {
                    process.StartInfo = psi;

                    process.OutputDataReceived += (s, e) =>
                    {
                        if (!string.IsNullOrWhiteSpace(e.Data))
                            OnLogReceived?.Invoke($"PIP: {e.Data}");
                    };
                    process.ErrorDataReceived += (s, e) =>
                    {
                        if (!string.IsNullOrWhiteSpace(e.Data))
                            OnLogReceived?.Invoke($"PIP LOG: {e.Data}");
                    };

                    process.Start();
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();

                    bool exited = process.WaitForExit(900000); // 15 dakika timeout

                    if (!exited)
                    {
                        process.Kill();
                        OnLogReceived?.Invoke("PIP: Kurulum zaman aşımına uğradı (15dk).");
                        OnFatalError?.Invoke("İnternet bağlantınız çok yavaş veya PyTorch sunucularına erişilemiyor. Lütfen manuel kurulum yapın.");
                        return false;
                    }
                    return process.ExitCode == 0;
                }
            }
            catch (Exception ex)
            {
                OnLogReceived?.Invoke($"PIP HATASI: {ex.Message}");
                return false;
            }
        }

        private void LogToUi(string message)
        {
            OnLogReceived?.Invoke(message);
        }
    }
}