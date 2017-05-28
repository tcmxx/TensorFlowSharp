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

            SetContentView(Resource.Layout.Main);

            TensorFlowSharp.Android.NativeBinding.Init();

            Log.Debug("TensorFlowSharp", "TF Version: " + TensorFlow.TFCore.Version);
        }
    }
}

