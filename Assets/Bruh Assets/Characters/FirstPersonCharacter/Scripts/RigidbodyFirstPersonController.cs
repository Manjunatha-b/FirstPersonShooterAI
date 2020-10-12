using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

namespace UnityStandardAssets.Characters.FirstPerson
{
    [RequireComponent(typeof (Rigidbody))]
    [RequireComponent(typeof (CapsuleCollider))]
    public class RigidbodyFirstPersonController : Agent
    {
        public GameObject bulletPrefab;
        public GameObject bulletSpawnLoc;
        [Serializable]


        public class MovementSettings
        {
            public float ForwardSpeed = 8.0f;   // Speed when walking forward
            public float BackwardSpeed = 4.0f;  // Speed when walking backwards
            public float StrafeSpeed = 4.0f;    // Speed when walking sideways
            public float RunMultiplier = 2.0f;   // Speed when sprinting
	        public KeyCode RunKey = KeyCode.LeftShift;
            public float JumpForce = 30f;
            public AnimationCurve SlopeCurveModifier = new AnimationCurve(new Keyframe(-90.0f, 1.0f), new Keyframe(0.0f, 1.0f), new Keyframe(90.0f, 0.0f));
            [HideInInspector] public float CurrentTargetSpeed = 8f;

            

#if !MOBILE_INPUT
            private bool m_Running;
#endif
            

            public void UpdateDesiredTargetSpeed(Vector2 input,float isrun)
            {
	            if (input == Vector2.zero) return;
				if (input.x > 0 || input.x < 0)
				{
					//strafe
					CurrentTargetSpeed = StrafeSpeed;
				}
				if (input.y < 0)
				{
					//backwards
					CurrentTargetSpeed = BackwardSpeed;
				}
				if (input.y > 0)
				{
					//forwards
					//handled last as if strafing and moving forward at the same time forwards speed should take precedence
					CurrentTargetSpeed = ForwardSpeed;
				}
#if !MOBILE_INPUT
	            if (isrun==1f)
	            {
		            CurrentTargetSpeed *= RunMultiplier;
		            m_Running = true;
	            }
	            else
	            {
		            m_Running = false;
	            }
#endif
            }

#if !MOBILE_INPUT
            public bool Running
            {
                get { return m_Running; }
            }
#endif
        }


        [Serializable]
        public class AdvancedSettings
        {
            public float groundCheckDistance = 0.01f; // distance for checking if the controller is grounded ( 0.01f seems to work best for this )
            public float stickToGroundHelperDistance = 0.5f; // stops the character
            public float slowDownRate = 20f; // rate at which the controller comes to a stop when there is no input
            public bool airControl; // can the user control the direction that is being moved in the air
            [Tooltip("set it to 0.1 or more if you get stuck in wall")]
            public float shellOffset; //reduce the radius by that ratio to avoid getting stuck in wall (a value of 0.1f is nice)
        }


        public Camera cam;
        public MovementSettings movementSettings = new MovementSettings();
        public MouseLook mouseLook = new MouseLook();
        public AdvancedSettings advancedSettings = new AdvancedSettings();


        private Rigidbody m_RigidBody;
        private CapsuleCollider m_Capsule;
        private float m_YRotation;
        private Vector3 m_GroundContactNormal;
        private bool m_Jump, m_PreviouslyGrounded, m_Jumping, m_IsGrounded;

        public Vector3 ResetPos;
        public Quaternion ResetRot;
        public Quaternion ResetCamRot;

        public GameObject[] enemies;
        public Vector3[] ResetEnemyPos;

        public Vector3 Velocity
        {
            get { return m_RigidBody.velocity; }
        }

        public bool Grounded
        {
            get { return m_IsGrounded; }
        }

        public bool Jumping
        {
            get { return m_Jumping; }
        }

        public bool Running
        {
            get
            {
 #if !MOBILE_INPUT
				return movementSettings.Running;
#else
	            return false;
#endif
            }
        }


        private void Start()
        {
            m_RigidBody = GetComponent<Rigidbody>();
            m_Capsule = GetComponent<CapsuleCollider>();
            mouseLook.Init (transform, cam.transform);
            ResetPos = m_RigidBody.transform.position;
            ResetRot = m_RigidBody.transform.rotation;
            ResetCamRot = cam.transform.localRotation;

            enemies = GameObject.FindGameObjectsWithTag("enemy");
            ResetEnemyPos = new Vector3[enemies.Length];
            for(int i = 0; i < enemies.Length; i++)
            {
                ResetEnemyPos[i] = new Vector3();
                ResetEnemyPos[i] = (enemies[i].transform.position);
            }
        }

        public void BulletHitEnemy()
        {
            AddReward(100f);
        }

        public void BulletHitWall()
        {

        }

       public override void Heuristic(float[] actionsOut)
        {
            actionsOut[0] = Input.GetAxis("Horizontal");
            actionsOut[1] = Input.GetAxis("Vertical");
            actionsOut[2] = !Input.GetKey(KeyCode.Mouse0)?0f:1f;
            actionsOut[3] = !Input.GetKey(KeyCode.Space)?0f:1f;
            actionsOut[4] = Input.GetAxis("Mouse X");
            actionsOut[5] = Input.GetAxis("Mouse Y");
            actionsOut[6] = !Input.GetKey(KeyCode.LeftShift)?0:1f;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            sensor.AddObservation(Time.timeScale);
            sensor.AddObservation(m_Jumping);
            sensor.AddObservation(this.transform.position);
            sensor.AddObservation(this.transform.rotation);
        }


        public override void OnEpisodeBegin()
        {
            base.OnEpisodeBegin();
            m_RigidBody.transform.position = ResetPos;
            m_RigidBody.transform.rotation = ResetRot;
            transform.localRotation = ResetRot;
            cam.transform.localRotation = ResetCamRot;
            mouseLook.Init(transform, cam.transform);
            for(int i = 0; i < enemies.Length; i++)
            {
                enemies[i].transform.position = ResetEnemyPos[i];
            }
        }

        public override void OnActionReceived(float[] vectorAction)
        {
            
            mouseLook.yInp = vectorAction[4];
            mouseLook.xInp = vectorAction[5];
            movementSettings.UpdateDesiredTargetSpeed(new Vector2(vectorAction[0], vectorAction[1]), vectorAction[6]);
            
            if (vectorAction[3] == 1f && !m_Jump)
            {
                m_Jump = true;
            }

            /////// My code from here on 
            if (vectorAction[2] == 1f)
            {
                GameObject bulletObject = Instantiate(bulletPrefab);
                bulletObject.transform.position = bulletSpawnLoc.transform.position;
                bulletObject.transform.forward = cam.transform.forward;
                bullet lmao = bulletObject.GetComponent<bullet>();
                lmao.assignParent(this);
            }


            GroundCheck();

            Vector2 input = new Vector2(vectorAction[0],vectorAction[1]);

            if ((Mathf.Abs(input.x) > float.Epsilon || Mathf.Abs(input.y) > float.Epsilon) && (advancedSettings.airControl || m_IsGrounded))
            {
                // always move along the camera forward as it is the direction that it being aimed at
                Vector3 desiredMove = cam.transform.forward * input.y + cam.transform.right * input.x;
                Time.timeScale = Mathf.Lerp(0.1f, 1, 2 * (Math.Abs(input.x) + Math.Abs(input.y)));
                desiredMove = Vector3.ProjectOnPlane(desiredMove, m_GroundContactNormal).normalized;
                desiredMove.x = desiredMove.x * movementSettings.CurrentTargetSpeed;
                desiredMove.z = desiredMove.z * movementSettings.CurrentTargetSpeed;
                desiredMove.y = desiredMove.y * movementSettings.CurrentTargetSpeed;
                if (m_RigidBody.velocity.sqrMagnitude <
                    (movementSettings.CurrentTargetSpeed * movementSettings.CurrentTargetSpeed))
                {
                    m_RigidBody.AddForce(desiredMove * SlopeMultiplier(), ForceMode.Impulse);
                }
            }

            if (m_IsGrounded)
            {
                m_RigidBody.drag = 5f;
                if (m_Jump)
                {
                    m_RigidBody.drag = 0f;
                    m_RigidBody.velocity = new Vector3(m_RigidBody.velocity.x, 0f, m_RigidBody.velocity.z);
                    m_RigidBody.AddForce(new Vector3(0f, movementSettings.JumpForce, 0f), ForceMode.Impulse);
                    m_Jumping = true;
                    Time.timeScale = 1;
                }

                if (!m_Jumping && Mathf.Abs(input.x) < float.Epsilon && Mathf.Abs(input.y) < float.Epsilon && m_RigidBody.velocity.magnitude < 1f)
                {
                    m_RigidBody.Sleep();
                    Time.timeScale = Mathf.Lerp(0.1f, 1, 2 * (Math.Abs(input.x) + Math.Abs(input.y)));
                }
            }
            else
            {
                m_RigidBody.drag = 0f;
                if (m_PreviouslyGrounded && !m_Jumping)
                {
                    StickToGroundHelper();

                }
            }
            m_Jump = false;
        }


        private void Update()
        {
            RotateView();
        }

        public void FixedUpdate()
        {
            AddReward(-0.01f);
            RequestDecision();
        }

        

        private float SlopeMultiplier()
        {
            float angle = Vector3.Angle(m_GroundContactNormal, Vector3.up);
            return movementSettings.SlopeCurveModifier.Evaluate(angle);
        }


        private void StickToGroundHelper()
        {
            RaycastHit hitInfo;
            if (Physics.SphereCast(transform.position, m_Capsule.radius * (1.0f - advancedSettings.shellOffset), Vector3.down, out hitInfo,
                                   ((m_Capsule.height/2f) - m_Capsule.radius) +
                                   advancedSettings.stickToGroundHelperDistance, Physics.AllLayers, QueryTriggerInteraction.Ignore))
            {
                if (Mathf.Abs(Vector3.Angle(hitInfo.normal, Vector3.up)) < 85f)
                {
                    m_RigidBody.velocity = Vector3.ProjectOnPlane(m_RigidBody.velocity, hitInfo.normal);
                }
            }
        }

        private void RotateView()
        {
            //avoids the mouse looking if the game is effectively paused
            if (Mathf.Abs(Time.timeScale) < float.Epsilon) return;

            // get the rotation before it's changed
            float oldYRotation = transform.eulerAngles.y;

            mouseLook.LookRotation (transform, cam.transform);

            if (m_IsGrounded || advancedSettings.airControl)
            {
                // Rotate the rigidbody velocity to match the new direction that the character is looking
                Quaternion velRotation = Quaternion.AngleAxis(transform.eulerAngles.y - oldYRotation, Vector3.up);
                m_RigidBody.velocity = velRotation * m_RigidBody.velocity;
            }
        }

        /// sphere cast down just beyond the bottom of the capsule to see if the capsule is colliding round the bottom
        private void GroundCheck()
        {
            m_PreviouslyGrounded = m_IsGrounded;
            RaycastHit hitInfo;
            if (Physics.SphereCast(transform.position, m_Capsule.radius * (1.0f - advancedSettings.shellOffset), Vector3.down, out hitInfo,
                                   ((m_Capsule.height/2f) - m_Capsule.radius) + advancedSettings.groundCheckDistance, Physics.AllLayers, QueryTriggerInteraction.Ignore))
            {
                m_IsGrounded = true;
                m_GroundContactNormal = hitInfo.normal;
            }
            else
            {
                m_IsGrounded = false;
                m_GroundContactNormal = Vector3.up;
            }
            if (!m_PreviouslyGrounded && m_IsGrounded && m_Jumping)
            {
                m_Jumping = false;
            }
        }
    }
}
