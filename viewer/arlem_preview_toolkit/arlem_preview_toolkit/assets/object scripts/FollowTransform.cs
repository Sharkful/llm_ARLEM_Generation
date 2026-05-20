using UnityEngine;

public class FollowTransform : MonoBehaviour
{
    private Transform target;
    private bool follow;

    private void Update()
    {
        if (follow)
        {
            transform.position = target.position;
        }
    }

    public void Follow(Transform followTransform)
    {
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.DEBUG,
            "Follow Called");
        
        target = followTransform;
        follow = true;
    }

    public void PauseFollow()
    {
        follow = false;
    }

    public void ResumeFollow()
    {
        follow = true;
    }
    
    public void StopFollow()
    {
        target = null;
        follow = false;
    }
}
