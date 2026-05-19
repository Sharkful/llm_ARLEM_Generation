using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit.Samples.StarterAssets;

public class labPlacementHandling : MonoBehaviour
{
    [SerializeField]
    private UnityEngine.InputSystem.InputActionReference activateTrigger;

    // Read only flag for when placement is done
    private bool _placed = false;
    public bool Placed
    {
        get { return _placed; }
    }
    private GameObject _anchor = null;
    public GameObject Anchor
    {
        get { return _anchor; }
    }

    private Vector3 _forward;
    public Vector3 Forward
    {
        get { return _forward;  }
    }

    public GameObject controller;
    

    private void Awake()
    {
        // Check that we have reference assigned
        if (activateTrigger == null)
        {
            LabLogger.Instance.InfoLog(
                GetType().ToString(),
                LabLogger.LogTag.ERROR,
                "Trigger action reference not set");
        }
    }

    private void Start()
    {
        if (controller == null)
        {
            LabLogger.Instance.InfoLog(
                GetType().ToString(),
                LabLogger.LogTag.DEBUG,
                "No reference to controller object, nothing to follow");
        }
    }

    private void OnEnable()
    {
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.TRACE,
            "lab Placement Handler enabled");
        _placed = false;
        activateTrigger.action.performed += OnTriggerPull;
    }

    private void OnDisable()
    {
        activateTrigger.action.performed -= OnTriggerPull;
    }

    private void Update()
    {
        // change transform to match the active controller position.
        transform.position = controller.transform.position;
        transform.eulerAngles = new Vector3(0, controller.transform.eulerAngles.y, 0);
    }

    private void OnTriggerPull(UnityEngine.InputSystem.InputAction.CallbackContext context)
    {
        // Create Anchor object
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.DEBUG,
            "Calling create anchor async");
        CreateAnchorAsync();
    }

    // Copied from Unity Docs: 
    // https://docs.unity3d.com/Packages/com.unity.xr.arfoundation@6.0/manual/features/anchors/aranchormanager.html
    async void CreateAnchorAsync()
    {
        // This is inefficient. You should re-use a saved reference instead.
        var manager = Object.FindAnyObjectByType<UnityEngine.XR.ARFoundation.ARAnchorManager>();

        var pose = new Pose(transform.position, transform.rotation);

        _forward = transform.position - Camera.main.transform.position;
        _forward.y = 0;

        var result = await manager.TryAddAnchorAsync(pose);
        if (result.status.IsSuccess())
        {
            // Store the reference to the instantiated object
            var anchorComponent = result.value;
            _anchor = anchorComponent.gameObject;

            // Do something with the newly created anchor.
            LabLogger.Instance.InfoLog(
                GetType().ToString(),
                LabLogger.LogTag.DEBUG,
                "Await add anchor returned success.");
            _placed = true;
        }
    }
}
