using UnityEngine;

public class LogThisObject : MonoBehaviour
{
    void Start()
    {
        // add to list of root objects
        registerWithLabLogger();
    }

    private void OnDestroy()
    {
        // remove from list of root objects
        unregisterWithLabLogger();
    }

    private void registerWithLabLogger()
    {
        LabLogger.Instance.RegisterObjectToLog(gameObject);
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.DEBUG,
            $"{name} tried to register as an Object To Log");
    }

    private void unregisterWithLabLogger()
    {
        LabLogger.Instance.UnregisterObjectToLog(gameObject);
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.DEBUG,
            $"{name} unregistered as an Object To Log");
    }
}
