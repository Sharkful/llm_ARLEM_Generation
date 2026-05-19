using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.SceneManagement;
using System.IO;
using System.Collections;
using Fusion;

public class LabManager : SimulationBehaviour, IPlayerJoined
{
    #region Variables
    [Header("Lab Filepath Data Object")]
    [SerializeField]
    private LabData sharedData;

    [Header("Lab Start")]
    [SerializeField]
    private bool colocation = false;
    [SerializeField]
    private InputActionReference controllerTriggerPull;
    [SerializeField]
    private GameObject controller = null;
    [SerializeField]
    private GameObject labOrigin;

    // Internal State
    private Vector3 clickPosition1=Vector3.zero, clickPosition2=Vector3.zero;
    private Vector3 anchorPosition;
    private Quaternion anchorRotation;
    private bool labAnchored = false;
    private bool labStarted = false;
    private string[] modules;
    private int moduleIndex = 0;
    private GameObject currentModuleObject;
    private ActivityModule currentModuleScript;

    private NetworkRunner runner;
    [SerializeField]
    private GameObject sphere;
    #endregion Variables

    #region Unity Methods
    private void Start()
    {
        StartCoroutine(ConnectToRunner());
    }

    private void OnDestroy()
    {
        if (runner != null)
        {
            runner.RemoveGlobal(this);
        }
    }
    #endregion Unity Methods

    #region Private Methods
    private void startLab()
    {
        if(labStarted)
        {
            LabLogger.Instance.InfoLog(
                GetType().ToString(),
                LabLogger.LogTag.DEBUG,
                "Tried to Start Lab, but it has already been started. Aborting");
            return;
        }

        labStarted = true;

        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.DEBUG,
            "Main Logic Started");

        // Start Loading necessary info in the background

        // Visual Debug tool, REMOVE FOR PRODUCTION
        sphere.GetComponent<MeshRenderer>().material.color = Color.green;

        // lab prep variables
        string debugLabZipFilename = "FullTransmissionLab.zip";
        FileInfo labZipFileInfo = new FileInfo(Path.Combine(
            Application.persistentDataPath,
            "lab_resources",
            debugLabZipFilename));
        const string DEBUG_URL = "https://cyberlearnar.cs.mtsu.edu/show_uploaded/";

        // force download of the lab files through download utility
        // reference login manager
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.DEBUG,
            "Creating lab zip directory");
        labZipFileInfo.Directory.Create();

        /*
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.DEBUG,
            "Attempting to start lab materials download from lab manager");
        DownloadUtility.Instance.DownloadAndExtractZip(
            Path.Combine(DEBUG_URL, debugLabZipFilename),
            labZipFileInfo.FullName,
            LabZipDownloadedAndExtracted,
            true);
        */
        // start lab directly using local files for now
        LabZipDownloadedAndExtracted(1);

        // Now check if we should start Colocation
        if (colocation)
        {
            // Start click tracking
            controllerTriggerPull.action.performed += OnTriggerPull;

            LabLogger.Instance.InfoLog(
                GetType().ToString(),
                LabLogger.LogTag.TRACE,
                $"Starting two point clicking anchor creation");
        }
    }

    private void SpawnModule()
    {
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.TRACE,
            $"SpawnModule() {moduleIndex}");
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.DEBUG,
            $"Scene Hierarchy before SpawnModule called:");
        LabLogger.Instance.LogSceneHierarchy();

        // Deserialize JSON to get the prefab name
        ActivityModuleData tmpData = new ActivityModuleData();
        JsonUtility.FromJsonOverwrite(modules[moduleIndex], tmpData);

        //Load prefab from resources
        GameObject tmpPrefab = (GameObject)Resources.Load($"Prefabs/{tmpData.prefabName}");

        // Instantiate prefab
        // Using our current location for now.
        currentModuleObject = Instantiate(tmpPrefab, labOrigin.transform);

        currentModuleScript = currentModuleObject.GetComponent<ActivityModule>();
        // Start the module
        currentModuleScript.TransmissionHost = runner.IsSharedModeMasterClient;
        currentModuleScript.TransmissionActivity = true;
        currentModuleScript.Initialize(modules[moduleIndex], moduleIndex);

        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.DEBUG,
            $"Scene Hierarchy after SpawnModule called:");
        LabLogger.Instance.LogSceneHierarchy();
        StartCoroutine(delayLogHierarchy());
    }
    #endregion Private Methods

    #region Event Handlers
    public void LabZipDownloadedAndExtracted(int rc)
    {
        if (rc == -1)
        {
            LabLogger.Instance.InfoLog(
                GetType().ToString(),
                LabLogger.LogTag.DEBUG,
                "Lab Resources Failed to Download");
        }
        else
        {
            LabLogger.Instance.InfoLog(
                GetType().ToString(),   
                LabLogger.LogTag.DEBUG,
                "Lab Resources Downloaded successfully");

        }

        string debugLabZipFilename = "FullTransmissionLab.zip";

        sharedData = new LabData();
        string labZipDirectory = Path.Combine(Application.persistentDataPath, "lab_resources");
        sharedData.labFilepath = new DirectoryInfo(Path.Combine(
            labZipDirectory,
            debugLabZipFilename.Substring(0, debugLabZipFilename.LastIndexOf("."))));

        // Find the json file
        FileInfo labJsonInfo = null;
        foreach (FileInfo file in sharedData.labFilepath.GetFiles())
        {
            LabLogger.Instance.InfoLog(
                GetType().ToString(),
                LabLogger.LogTag.DEBUG,
                $"Checking file {file.Name}, in Directory {file.DirectoryName}");

            if (file.Name.EndsWith(".json"))
            {
                labJsonInfo = file;
                break;
            }
        }

        // Check if we found the lab json
        if (labJsonInfo == null)
        {
            LabLogger.Instance.InfoLog(
                GetType().ToString(),
                LabLogger.LogTag.ERROR,
                $"Could not find lab json while starting lab scene");

            // Go back to login scene
            StartCoroutine(LoadLoginScene());
        }

        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.TRACE,
            $"Attempting to read lab json into labdataobject");
        // Create and initialize the lab data object
        LabDataObject labData = new LabDataObject();
        string json_content = File.ReadAllText(labJsonInfo.FullName);
        JsonUtility.FromJsonOverwrite(json_content, labData);

        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.TRACE,
            $"Attempting to get list of modules");
        // Get json strings describing each module
        modules = labData.ActivityModules;

        // Start initializing Catalogue
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.TRACE,
            $"Attempting to initialize Media Catalogue");
        MediaCatalogue mc = GetComponent<MediaCatalogue>();
        mc.enabled = true;
        mc.InitializeCatalogue(sharedData.labFilepath);

        // Await mc loading
        StartCoroutine(await_mc(mc));
    }

    public void OnTriggerPull(InputAction.CallbackContext context)
    {
        // Handle first click
        if (clickPosition1 == Vector3.zero)
        {
            // Record position
            clickPosition1 = controller.transform.position;
            // place marker on click location
            var marker1 = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            marker1.transform.position = clickPosition1;
            marker1.transform.localScale = Vector3.one * 0.5f;
        }
        else if (clickPosition2 == Vector3.zero)
        {
            // Record position
            clickPosition2 = controller.transform.position;
            // place marker on click location
            var marker2 = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            marker2.transform.position = clickPosition1;
            marker2.transform.localScale = Vector3.one * 0.5f;
            // calculate and store the anchor position
            anchorPosition = (clickPosition1 + clickPosition2) / 2;
            anchorRotation = Quaternion.LookRotation(clickPosition2 - clickPosition1);
            labAnchored = true;
            // stop responding to clicks
            controllerTriggerPull.action.performed -= OnTriggerPull;
            // Move on to the next step
            // If we are the master client, spawn objects
            if (Runner.IsSharedModeMasterClient)
            {
                // Do something here that enables moving on to the next step
                // Move scene root to the new position, and enable calling startLab
                labOrigin.transform.SetPositionAndRotation(anchorPosition, anchorRotation);
                labAnchored = true;
            }
            else // we are the peer
            {
                // Move scene root to the new position,
                // ask for control of object
                labOrigin.GetComponent<NetworkObject>().RequestStateAuthority();
                labOrigin.transform.SetPositionAndRotation(anchorPosition, anchorRotation);
                labAnchored = true;
                labOrigin.GetComponent<NetworkObject>().ReleaseStateAuthority();
            }
        }
    }

    public void ModuleComplete(int direction)
    {
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.DEBUG,
            $"Scene Hierarchy before ModuleComplete called:");
        LabLogger.Instance.LogSceneHierarchy();

        // make sure direction is 1 or -1
        direction /= Mathf.Abs(direction);
        // Apply direction to index
        moduleIndex += direction;
        // Make sure we have a valid index, or it goes over the last index
        moduleIndex = Mathf.Clamp(moduleIndex, 0, modules.Length);

        // clean up previous module object.
        GameObject.Destroy(currentModuleObject);
        currentModuleObject = null;
        currentModuleScript = null;

        // Destroy all objects under the anchor
        foreach (Transform t in labOrigin.transform)
        {
            GameObject.Destroy(t.gameObject);
        }

        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.DEBUG,
            $"Scene Hierarchy after ModuleComplete called, before SpawnModule():");
        LabLogger.Instance.LogSceneHierarchy();

        // Check if we go over the last index
        if (moduleIndex < modules.Length)
        {
            //SpawnModule();
            StartCoroutine(SpawnModuleDelay());
        }
        else
        {
            // Then end the lab, go to login screen
            StartCoroutine(LoadLoginScene());
        }
    }

    public void PlayerJoined(PlayerRef player)
    {
        if( player == Runner.LocalPlayer)
        {
            LabLogger.Instance.InfoLog(
                GetType().ToString(),
                LabLogger.LogTag.TRACE,
                $"Local player joined event received");
            startLab();
        }
    }
    #endregion EventHandlers

    #region Coroutines
    private IEnumerator ConnectToRunner()
    {
        // Initialize to null
        runner = null;
        // Repeat search while still not found
        while (runner == null)
        {
            // Iterate through each possible instance of network runner
            foreach (var possibleRunner in NetworkRunner.Instances)
            {
                // check that the reference is valid, and that the runner is active
                if (possibleRunner != null && possibleRunner.IsRunning)
                {
                    // we found it
                    runner = possibleRunner;
                    break;
                }
            }
            // If we still haven't found it, wait a bit before checking again
            if (runner == null)
            {
                yield return new WaitForSeconds(0.5f);
            }
        }

        // Now that we have the runner reference, register with it
        if (this != null)
        {
            sphere.GetComponent<MeshRenderer>().material.color = Color.yellow;
            runner.AddGlobal(this);
            LabLogger.Instance.InfoLog(
                GetType().ToString(),
                LabLogger.LogTag.TRACE,
                $"Lab Manager connected to runner {runner.name}");

            // Check for our custom network object provider
            var checkNOP = runner.GetComponent<ARLabCustomNetwokObjectProvider>();
            if (checkNOP == null)
            {
                LabLogger.Instance.InfoLog(
                    GetType().ToString(),
                    LabLogger.LogTag.ERROR,
                    $"Could not find custom object provider on active runner");
            }
            else
            {
                LabLogger.Instance.InfoLog(
                    GetType().ToString(),
                    LabLogger.LogTag.ERROR,
                    $"Found our custom object provider on the active runner!");
            }

            // Check if local player connected while we waited for the runner
            if (runner.LocalPlayer.IsRealPlayer)
            {
                LabLogger.Instance.InfoLog(
                    GetType().ToString(),
                    LabLogger.LogTag.TRACE,
                    $"Local Player already joined! starting lab");
                startLab();
            }
        }
    }

    private IEnumerator await_mc(MediaCatalogue mc)
    {
        yield return new WaitUntil(() => mc.DoneLoadingAssets);

        // Then spawn the module if Placement is done
        if (labAnchored)
        {
            try
            {
                StartCoroutine(SpawnModuleDelay());
            }
            catch (System.Exception ex)
            {
                LabLogger.Instance.InfoLog(
                    GetType().ToString(),
                    LabLogger.LogTag.ERROR,
                    ex.ToString());
            }
        }
        else // Lab not yet anchored, wait
        {
            StartCoroutine(await_anchor());
        }
    }

    private IEnumerator await_anchor()
    {
        yield return new WaitUntil(() => labAnchored);

        // start lab
        try
        {
            StartCoroutine(SpawnModuleDelay());
        }
        catch (System.Exception ex)
        {
            LabLogger.Instance.InfoLog(
                GetType().ToString(),
                LabLogger.LogTag.ERROR,
                ex.ToString());
        }
    }

    private IEnumerator SpawnModuleDelay()
    {
        // yield return null;
        yield return new WaitForSeconds(2.0f);
        try
        {
            SpawnModule();
        }
        catch (System.Exception ex)
        {
            LabLogger.Instance.InfoLog(
                GetType().ToString(),
                LabLogger.LogTag.ERROR,
                ex.ToString());
        }
    }

    private IEnumerator delayLogHierarchy()
    {
        yield return new WaitForSecondsRealtime(1.0f);
        LabLogger.Instance.LogSceneHierarchy();
    }

    private IEnumerator LoadLoginScene()
    {
        AsyncOperation sceneLoading = SceneManager.LoadSceneAsync(0);
        yield return new WaitUntil(() => sceneLoading.isDone);
    }
    #endregion Coroutines
}
