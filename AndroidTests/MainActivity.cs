using Android.App;
using Android.Widget;
using Android.OS;
using Android.Util;

namespace AndroidTests
{
    [Activity(Label = "AndroidTests", MainLauncher = true, Icon = "@drawable/icon")]
    public class MainActivity : Activity
    {
        protected override void OnCreate(Bundle bundle)
        {
            base.OnCreate(bundle);

            TensorFlowSharp.Android.NativeBinding.Init();

            SetContentView(Resource.Layout.Main);

            Button bt = FindViewById<Button>(Resource.Id.MyButton);
            bt.Click += Bt_Click;
        }

        private void Bt_Click(object sender, System.EventArgs e)
        {
            Log.Debug("TensorFlowSharp", "TF Version: " + TensorFlow.TFCore.Version);
            ((Button)sender).Text = $"Tensorflow Version: {TensorFlow.TFCore.Version}";
        }
    }
}