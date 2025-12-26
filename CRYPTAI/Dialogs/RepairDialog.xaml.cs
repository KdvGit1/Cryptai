using System.Windows;
using Application = System.Windows.Application;

namespace CRYPTAI
{
    public partial class RepairDialog : Window
    {
        public bool IsRepairRequested { get; private set; } = false;

        public RepairDialog()
        {
            InitializeComponent();
        }

        private void BtnRepair_Click(object sender, RoutedEventArgs e)
        {
            IsRepairRequested = true;
            this.Close();
        }

        private void BtnCancel_Click(object sender, RoutedEventArgs e)
        {
            IsRepairRequested = false;
            this.Close();
            Application.Current.Shutdown();
        }
    }
}