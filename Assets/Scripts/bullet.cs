using UnityEngine;
using UnityStandardAssets.Characters.FirstPerson;

public class bullet : MonoBehaviour
{
    public float speed = 60f;
    public ParticleSystem bulletFlash;
    Vector3 previouspos;
    Vector3 startpos;
    bool hit;
    RigidbodyFirstPersonController parent;
    
    // Start is called before the first frame update
    void Start()
    {
        previouspos = transform.position;
        startpos = transform.position;
        hit = false;
    }

    public void assignParent(RigidbodyFirstPersonController p1)
    {
        parent = p1;
    }

    void Update()
    {
        transform.position += transform.forward * speed * Time.deltaTime;
    }
    // Update is called once per frame
    void FixedUpdate()
    {
        RaycastHit herebruh;

        Ray waw = new Ray(previouspos, (this.transform.position - previouspos).normalized);
        if (Physics.Raycast(waw, out herebruh, (transform.position - previouspos).magnitude) && !hit)
        {
            ParticleSystem bruh = Instantiate(bulletFlash);
            bruh.transform.position = herebruh.point;
            bruh.transform.forward = herebruh.normal;
            if (herebruh.transform.tag == "enemy")
            {
                herebruh.transform.gameObject.transform.position = new Vector3()
                parent.loghit();
            }
            Destroy(this.gameObject);
            hit = true;
        }
        if (Vector3.Distance(startpos, transform.position) > 250f)
        {
            
            Destroy(this.gameObject);
        }
        previouspos = transform.position;
    }
}
