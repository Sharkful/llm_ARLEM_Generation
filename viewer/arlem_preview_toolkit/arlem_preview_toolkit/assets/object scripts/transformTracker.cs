using UnityEngine;
using System.Collections;

public class transformTracker : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        StartCoroutine(DelayedPositionReport());
    }

    private IEnumerator DelayedPositionReport()
    {
        while(true)
        {
            yield return new WaitForSeconds(5.0f);
            LabLogger.Instance.InfoLog(
                GetType().ToString(),
                LabLogger.LogTag.DEBUG,
                $"{name} | Parent : {transform.parent} | Pos : {transform.position.ToString("F3")} | LPos : {transform.localPosition.ToString("F3")}");
        }
    }
}
