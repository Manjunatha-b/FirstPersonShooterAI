using UnityEngine;

public class Gun : MonoBehaviour
{
    // Start is called before the first frame update
    public GameObject cam;
    // Update is called once per frame
    void Update()
    {
        this.transform.right = -cam.transform.forward;
    }
}
