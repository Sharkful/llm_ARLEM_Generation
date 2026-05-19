using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Assertions;
using UnityEngine.SceneManagement;
using System.Collections;
using System.Collections.Generic;
using System.IO;

public class LoginManager : MonoBehaviour
{
    #region Variables
    // Enums
    private enum EndpointType
    {
        debug,
        production
    }
    private enum State
    {
        STARTUP,
        INTRODUCTION,
        LOGIN,
        LAB_SELECTION
    }

    // Inspector Variables
    [Header("Controller Reference")]
    [SerializeField]
    private GameObject controllerObject;

    [Header("Server Interactions")]
    [SerializeField]
    private EndpointType endpointType;
    [SerializeField]
    [Tooltip("Whether we always try to redownload files. If false, uses local copies if they can be found")]
    private bool forceDownloads = false;
    [SerializeField]
    private string debugLabListFilename = "dev_labs_list.json";
    [SerializeField]
    private string debugLabZipFilename = "FullTransmissionLab.zip";

    [Header("Introduction Animation")]
    [SerializeField]
    private GameObject mt_logo;

    [Header("Login")]
    [SerializeField]
    private GameObject loginUI;
    [SerializeField]
    private Text pinInput;
    [SerializeField]
    private GameObject loading;
    [SerializeField]
    private Authenticate auth;
    [SerializeField]
    private float authTimeout = 15f;

    [Header("LabSelection")]
    [Tooltip("Used to clone and make new lab selection buttons")]
    [SerializeField]
    private GameObject labTemp;
    [SerializeField]
    [Tooltip("UI Gameobject containing list of labs as buttons")]
    private GameObject labOptions;
    [SerializeField]
    [Tooltip("UI grid object organizing lab buttons")]
    private GameObject labOptionsGrid;
    [SerializeField]
    private float labListParseTimeout = 15f;
    private List<GameObject> labListButtons = new List<GameObject>();
    private List<LabInfo> labInfoList = new List<LabInfo>();
    private LabInfo selectedLab;

    [Header("Object To Store Lab Filepath")]
    [SerializeField]
    private LabData sharedData;

    // Internal State
    private State currentState = State.STARTUP;
    private bool labListReady;

    // Base URLs to download from
    const string LABS_URL = "https://cyberlearnar.cs.mtsu.edu/labs";
    const string LAB_ZIP_BASE_URL = "https://cyberlearnar.cs.mtsu.edu/show_uploaded/lab_";
    const string DEBUG_URL = "https://cyberlearnar.cs.mtsu.edu/show_uploaded/";

    // Download File Locations
    private FileInfo allLabsFileInfo;
    private FileInfo labZipFileInfo;
    private string labZipDirectory;
    #endregion Variables

    #region Unity Methods
    private void Awake()
    {
        // Fail Fast
        Assert.IsNotNull(mt_logo);

        Assert.IsNotNull(loginUI);
        Assert.IsNotNull(pinInput);
        Assert.IsNotNull(loading);
        Assert.IsNotNull(auth);

        Assert.IsNotNull(labTemp);
        Assert.IsNotNull(labOptions);
        Assert.IsNotNull(labOptionsGrid);

        // Initialize Download Directories
        allLabsFileInfo = new FileInfo(                                  // Create FileInfo from filepath
            Path.Combine(Application.persistentDataPath, "login", "All_Labs.json" )
        );// lab list file name
        allLabsFileInfo.Directory.Create();                              // Create the directolabZipDirectoryry if not there already.
        labZipDirectory = Path.Combine(Application.persistentDataPath, "lab_resources");
    }


    void Start()
    {
        // Start by checking SO to see if we have already logged in
        if (sharedData.LoginComplete)
        {
            // then parse labs, build lab list, then go to lab list
            parseLabs();

            changeStateTo(State.LAB_SELECTION);
        }
        else
        {
            try
            {
                // Download the list of available labs
                if (endpointType == EndpointType.production)
                    DownloadUtility.Instance.DownloadFile(LABS_URL, allLabsFileInfo.FullName, labListDownloaded, !forceDownloads);
                else if (endpointType == EndpointType.debug)
                {
                    DownloadUtility.Instance.DownloadFile(
                        Path.Combine(DEBUG_URL, debugLabListFilename),
                        allLabsFileInfo.FullName,
                        labListDownloaded,
                        !forceDownloads);
                }

                // Start the logo animation
                changeStateTo(State.INTRODUCTION);
            }
            catch (System.Exception e)
            {
                LabLogger.Instance.InfoLog(
                    GetType().ToString(),
                    LabLogger.LogTag.DEBUG,
                    $"{e}");
            }
        }
    }
    #endregion Unity Methods

    #region Private Methods
    private void changeStateTo(State newState)
    {
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.TRACE,
            $"Changing from {currentState} to {newState}");
        
        if (newState == currentState)
            return;

        // Handle leaving the old state
        switch(currentState)
        {
            case State.INTRODUCTION:
                {
                    // Disable logo 
                    mt_logo.SetActive(false);
                    break;
                }
            case State.LOGIN:
                {
                    // Disable and reset login UI
                    loginUI.SetActive(false);
                    loading.SetActive(false);
                    pinInput.text = "";
                    break;
                }
            case State.LAB_SELECTION:
                {
                    if (labListReady)
                    {
                        labOptions.SetActive(false);
                    }
                    else
                    {
                        StopCoroutine(AwaitLabList());
                    }
                    break;
                }
            default:
                break;
        }

        // Handle Entering New state
        switch(newState)
        {
            case State.INTRODUCTION:
                {
                    // Start the MT logo animation
                    StartCoroutine(SetupMTLogoAnimation());
                    break;
                }
            case State.LOGIN:
                {
                    // Enable login UI
                    loginUI.SetActive(true);
                    break;
                }
            case State.LAB_SELECTION:
                {
                    if (labListReady)
                    {
                        labOptions.SetActive(true);
                        if(labListButtons.Count == 0)
                            generateLabListUI();
                    }
                    else 
                    {
                        loading.SetActive(true);
                        StartCoroutine(AwaitLabList());
                    }
                    break;
                    //nothing
                }
            default:
                break;
        }

        // Update the state
        currentState = newState;
    }

    private void labListDownloaded(int rc)
    {
        if(rc == 0) // File Successfully downloaded
        {
            // Generate lab list with titles, ids, and deccriptions
            parseLabs();
        }
        else // Download Failed
        {
            LabLogger.Instance.InfoLog(
                GetType().ToString(),
                LabLogger.LogTag.ERROR,
                "Lab list failed to download");
        }
    }

    private void parseLabs()
    {
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.TRACE,
            "ParseLabs()");

        // Trim leading and trailing [{}]
        string labsString = allLabsFileInfo.OpenText().ReadToEnd().Trim(new char[] { '[', '{', '}', ']' });
        // Split by },{ which only occurs between labs in the list
        string[] labElements = labsString.Split(new string[] { "},{" }, System.StringSplitOptions.None);

        foreach (string lab in labElements)
        {
            LabInfo tmpLabInfo = new LabInfo();

            // Find description
            string descSearch = "\"lab_description\":\"";
            int descLoc = lab.IndexOf(descSearch) + descSearch.Length;
            tmpLabInfo.description = lab.Substring(descLoc, lab.IndexOf("\"", descLoc) - descLoc);

            // Find id
            string idSearch = "\"lab_id\":";
            int idLoc = lab.IndexOf(idSearch) + idSearch.Length;
            tmpLabInfo.id = lab.Substring(idLoc, lab.IndexOf(",", idLoc) - idLoc);
            
            // Find name
            string nameSearch = "\"lab_title\":\"";
            int nameLoc = lab.IndexOf(nameSearch) + nameSearch.Length;
            tmpLabInfo.name = lab.Substring(nameLoc, lab.IndexOf("\"", nameLoc) - nameLoc);

            labInfoList.Add(tmpLabInfo);
        }

        // Log the successful parse
        string labNames = "";
        foreach (LabInfo lab in labInfoList)
            labNames += lab.name + ", ";
        
        // Set flag that the lab list is parsed and ready to be used
        labListReady = true;
    }

    private void attemptLogin(string pin)
    {
        // If auth isn't set up yet, then wait
        if (!auth.Ready)
        {
            LabLogger.Instance.InfoLog(
                GetType().ToString(),
                LabLogger.LogTag.TRACE,
                $"Auth not yet ready, waiting");
            auth.OnReady += () => AuthReady(pin);
            return;
        }

        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.TRACE,
            $"Attempting Login with pin {pin}");
        // enable loading wheel while authenticating
        loading.SetActive(true);
        bool success = auth.AuthenticatePin(pin);
        loading.SetActive(false);

        
        if (success)
        {
            changeStateTo(State.LAB_SELECTION);
        }
    }

    private void generateLabListUI()
    {
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.TRACE,
            "GenerateLabListUI()");

        // Generate 5 lab buttons, or less if the list of available labs is short
        int i = 0;
        while (i < 5 && i < labInfoList.Count)
        {
            // Create lab ui object and position it
            GameObject tmpLabUI = Instantiate(labTemp, labOptionsGrid.transform);
            //tmpLabUI.transform.position += .42f * (i % 2) * loginUI.transform.right + new Vector3(0, -.15f * (i / 2), 0);

            // Add listener to button to transition to next stage
            int i_copy = i; // This is done b/c of some anonymous function variable capturing loop variable technicalities
            tmpLabUI.GetComponentInChildren<Button>().onClick.AddListener(() => LabSelected(labInfoList[i_copy].id));

            // Setup object, add to list
            tmpLabUI.transform.Find("Lab Title").GetComponent<Text>().text = labInfoList[i].name;
            tmpLabUI.transform.Find("Lab Description").GetComponent<Text>().text = labInfoList[i].description;
            tmpLabUI.name = labInfoList[i].name;
            tmpLabUI.SetActive(true);
            labListButtons.Add(tmpLabUI);

            i++;
        }

        // Create an exit button using the lab button template
        GameObject exitUI = Instantiate(labTemp, labOptionsGrid.transform);
        exitUI.transform.position += .42f * (i % 2) * loginUI.transform.right + new Vector3(0, -.15f * (i / 2), 0);

        // Add listiner to end application
        exitUI.GetComponentInChildren<Button>().onClick.AddListener(() => ExitSelected());

        // Set title and description
        exitUI.transform.Find("Lab Title").GetComponent<Text>().text = "Exit";
        exitUI.transform.Find("Lab Description").GetComponent<Text>().text = "Close ARLabs";

        exitUI.name = "ExitButton";
        exitUI.SetActive(true);
        labListButtons.Add(exitUI);
    }

    private void LabSelected(string labId)
    {
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.TRACE,
            "LabSelected()");

        selectedLab = labInfoList.Find(x => x.id == labId);

        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.DEBUG,
            $"Lab Selected: {selectedLab.id}, {selectedLab.name}");

        // Download the Zip file containing all lab resources
        if (endpointType == EndpointType.production)
        {
            labZipFileInfo = new FileInfo(Path.Combine(
                Application.persistentDataPath,
                "lab_resources",
                selectedLab.id + ".zip"));
            labZipFileInfo.Directory.Create();

            DownloadUtility.Instance.DownloadAndExtractZip(
                Path.Combine(LAB_ZIP_BASE_URL, selectedLab.id + ".zip"),
                labZipFileInfo.FullName,
                LabZipDownloadedAndExtracted,
                !forceDownloads);
        }
        else if (endpointType == EndpointType.debug)
        {
            labZipFileInfo = new FileInfo(Path.Combine(
                Application.persistentDataPath,
                "lab_resources",
                debugLabZipFilename));
            labZipFileInfo.Directory.Create();

            DownloadUtility.Instance.DownloadAndExtractZip(
                Path.Combine(DEBUG_URL, debugLabZipFilename),
                labZipFileInfo.FullName,
                LabZipDownloadedAndExtracted,
                !forceDownloads);
        }
    }

    private void LabZipDownloadedAndExtracted(int rc)
    {
        // Download and extraction were a success
        if (rc == 0)
        {
            if (endpointType == EndpointType.debug)
                LabStart(new DirectoryInfo(Path.Combine(
                    labZipDirectory,
                    debugLabZipFilename.Substring(0, debugLabZipFilename.LastIndexOf(".")))));
            else if (endpointType == EndpointType.production)
                LabStart(new DirectoryInfo(Path.Combine(
                    labZipDirectory,
                    selectedLab.id)));
        }
        else // Download failed
            LabLogger.Instance.InfoLog(
                GetType().ToString(),
                LabLogger.LogTag.ERROR,
                $"LabZip failed to download. Stopping Program");
    }

    private void LabStart(DirectoryInfo labResourcesFolder)
    {
        // Write directory Info to scriptable object to be stored and used in the next lab scene
        sharedData.labFilepath = labResourcesFolder;
        sharedData.LoginComplete = false;
        sharedData.ControllerObject = controllerObject;

        // Start loading the next scene
        StartCoroutine(LoadLabScene());
    }

    private void LogSubmitted()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
    }
    #endregion Private Methods

    #region Public Callbacks
    public void AuthReady(string pin)
    {
        attemptLogin(pin);
    }

    /// <summary>
    /// Called by the enter key on the keyboard, start authentication
    /// </summary>
    public void PinEntered()
    {
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.TRACE,
            "PinEntered()");
        attemptLogin(pinInput.text);
    }

    public void GuestLogin()
    {
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.TRACE,
            "GuestLogin()");
        pinInput.text = "000000";
        attemptLogin(pinInput.text);
    }

    public void ExitSelected()
    {
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.TRACE,
            "ExitSelected()");
        LabLogger.Instance.SubmitLog(LogSubmitted);
    }

    #endregion Public Callbacks

    #region Coroutines
    private IEnumerator SetupMTLogoAnimation()
    {
        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.TRACE,
            "SetupMTLogoAnimation()");

        // While the main camera hasn't moved from the origin, wait for it to move (sync to headset position)
        yield return new WaitUntil(() => Camera.main.transform.position != Vector3.zero);

        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.DEBUG,
            "MTLogo Animation, passed waituntil for camera to move");

        // Once the camera is synced to the headset position, Place the mt logo
        mt_logo.transform.position = Camera.main.transform.position + Camera.main.transform.forward * 1;

        // Rotate the logo to face the camera
        mt_logo.transform.eulerAngles = new Vector3(0, Camera.main.transform.eulerAngles.y, 0);
        
        // Enable the Logo with the animator on it
        mt_logo.SetActive(true);

        // Wait for animation to finish
        yield return new WaitUntil(
            () => mt_logo.GetComponentInChildren<Animator>().GetCurrentAnimatorStateInfo(0).IsName("Idle"));

        LabLogger.Instance.InfoLog(
            GetType().ToString(),
            LabLogger.LogTag.DEBUG,
            "MTLogoAnimation passed 2nd waituntil, animation is at Idle, should move to login");

        // start login
        changeStateTo(State.LOGIN);
    }

    private IEnumerator AwaitLabList()
    {
        float startTime = Time.time;
        yield return new WaitUntil(() => labListReady || (Time.time - startTime) > labListParseTimeout);

        loading.SetActive(false);

        labOptions.SetActive(true);
        generateLabListUI();
    }

    private IEnumerator LoadLabScene()
    {
        AsyncOperation sceneLoading = SceneManager.LoadSceneAsync(1);
        yield return new WaitUntil(() => sceneLoading.isDone);
    }
    #endregion Coroutines
}
