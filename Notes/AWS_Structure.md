
## AWS Structure

```
AWS Organization (optional, usually company-level)
│
├── AWS Account: Staging
│   Account ID: 9770-0576-3657
│   Alias: edriving-mentorstaging
│
│   ├── Root user
│   │      └── Created automatically with the AWS account
│   │
│   ├── IAM Users
│   │      ├── Ganeshrajesh
│   │      │     ├── Console password + MFA
│   │      │     ├── Access Key + Secret Key
│   │      │     └── Permissions
│   │      │           ├── directly attached policies
│   │      │           └── IAM Group policies
│   │      │
│   │      └── Other IAM users...
│   │
│   ├── IAM Groups
│   │      ├── Developers
│   │      │      └── Policies
│   │      └── Admins
│   │             └── Policies
│   │
│   ├── IAM Roles
│   │      ├── DeveloperRole
│   │      ├── AdminRole
│   │      ├── EC2Role
│   │      └── LambdaRole
│   │
│   └── AWS Resources
│          ├── EC2
│          ├── DynamoDB
│          ├── S3
│          ├── Lambda
│          └── etc.
│
└── AWS Account: Production
       Account ID: different
       └── its own IAM users/roles/resources
```

### Permissions = Policies

### IAM ROLE VS GROUP

| AWS IAM object | Holds policies? | Who/what can use it? |
| --- | --- | --- |
| IAM User | ✅ | One IAM identity |
| IAM Group | ✅ | IAM users only |
| IAM Role | ✅ | Users, SSO identities, AWS services like EC2/Lambda, other AWS accounts, etc. |
| IAM Policy | N/A | Defines the actual permissions |

```
             IAM ROLE
            /        \
           /          \
 Trust Policy      Permission Policies
      │                    │
      │                    │
Who can assume it?    What can it do?
      │                    │
      ▼                    ▼
     EC2             S3 GetObject
                     DynamoDB GetItem
                     etc.
```
- IAM ROLE: No one can direcly assume it unless its mentioned in Trust Policy
- An IAM role essentially answers two questions:
    - Who is allowed to assume me? → Trust policy
    - What can I do after being assumed? → Permission policies
- IAM ROLES are for Users or Services to access resoureces for Temporory
- IAM ROLE session: Temp Session: Admin decided session time for roles or default time
```
IAM Role (exists permanently)
   │
   ├── Trust policy → WHO can assume it
   └── Permission policies → WHAT it can do
             │
             ▼
       Someone assumes role
             │
             ▼
      Temporary session
             │
      ┌──────┼────────┐
      ▼      ▼        ▼
 AccessKey SecretKey SessionToken
             │
             ▼
       Session expires
```