using UnityEngine;

public class SimpleZRotation : MonoBehaviour
{
    [SerializeField]
    private float rotSpeed = 3.14159f * 0.75f; // default to 3/4 turn per second
    
    // Update is called once per frame
    void Update()
    {
        transform.Rotate(0.0f, 0.0f, rotSpeed * Time.deltaTime);
    }
}
