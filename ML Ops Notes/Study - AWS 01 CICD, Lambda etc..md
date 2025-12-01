

# Production ML Inference on AWS Lambda: A Complete MLOps Study Guide

**Outcome:** Production-ready ML inference API with full CI/CD pipeline  
**Final Stats:** 25 tests, 97% coverage, $0.00/month cost, fully automated deployment

---

## Overview:

A **serverless machine learning inference system** that demonstrates enterprise-grade MLOps practices:

```
Developer Push → GitHub Actions → Automated Tests → SAM Deploy → AWS Lambda
                                                                      ↓
                                                            Live ML Predictions
```

**Core Technologies:**

- **ML:** scikit-learn Random Forest (Iris classification)
- **Backend:** Python 3.11, FastAPI-style handler
- **Testing:** pytest with 97% coverage
- **Infrastructure:** AWS SAM (Serverless Application Model)
- **CI/CD:** GitHub Actions
- **Deployment:** AWS Lambda (512MB, serverless)

**Key Patterns Demonstrated:**

- Clean 3-layer architecture (handler → service → loader)
- Singleton pattern for cold start optimization
- Environment-aware path resolution
- Infrastructure as Code
- Automated testing gates
- Secret management

---

## Table of Contents

1. [Phase 1: Environment & Model Training]
2. [Phase 2: The Path Resolution Challenge]
3. [Phase 3: Three-Layer Architecture]
4. [Phase 4: The Singleton Pattern]
5. [Phase 5: AWS SAM Deep Dive]
6. [Phase 6: Lambda Environment Mysteries]
7. [Phase 7: GitHub Actions Pipeline]
8. [Key Learnings & Patterns]

---

## Phase 1: Environment & Model Training

### The Foundation: Why Pre-train?

**Critical Concept:** Lambda is for **serving**, not training.

```python
# WRONG: Training in Lambda
def lambda_handler(event, context):
    X_train, y_train = load_data()  # ❌ Too slow
    model = train_model(X_train, y_train)  # ❌ Takes minutes
    return predict(model, event)

# RIGHT: Load pre-trained model
def lambda_handler(event, context):
    model = get_cached_model()  # ✅ Milliseconds
    return predict(model, event)
```

**Why this matters:**

- Lambda timeout: 30 seconds maximum
- Training takes: Minutes to hours
- Inference takes: Milliseconds
- Lambda charges per millisecond of execution

> **Pattern:** Separate training (one-time, local) from inference (repeated, serverless).

---

### Model Training Script Architecture

**File:** `train_model.py`

```python
# Conceptual structure
def main():
    # 1. Load data
    X, y, feature_names, target_names = load_iris_data()
    
    # 2. Split with reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42  # ← Ensures identical splits every run
    )
    
    # 3. Train
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42  # ← Deterministic model initialization
    )
    model.fit(X_train, y_train)
    
    # 4. Evaluate before saving
    accuracy = model.score(X_test, y_test)  # Must be >95% to be useful
    
    # 5. Serialize
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)  # Binary format, ~10KB
```

**Key Decision:** `random_state=42` everywhere.

**Why reproducibility matters:**

```python
# Without random_state:
Run 1: accuracy = 0.967
Run 2: accuracy = 0.933
Run 3: accuracy = 0.950
# Tests become flaky!

# With random_state=42:
Run 1: accuracy = 0.967
Run 2: accuracy = 0.967
Run 3: accuracy = 0.967
# Tests are deterministic ✅
```

---

## Phase 2: The Path Resolution Challenge

### The Problem: Where Am I?

**Context:** Code runs in multiple environments:

- Local development: `D:\...\CICD_ML_Infr_Lambda\`
- Lambda production: `/var/task/`
- GitHub Actions: `/home/runner/work/.../CICD_ML_Infr_Lambda/`

**Common antipattern:**

```python
# ❌ FRAGILE - breaks in Lambda
import os
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
```

**Why this fails:**

```
Local:  src/model_loader.py
        ↓ parent
        src/
        ↓ parent  
        CICD_ML_Infr_Lambda/  ← Found!

Lambda: /var/task/src/model_loader.py
        ↓ parent
        /var/task/src/
        ↓ parent
        /var/task/  ← NOT named "CICD_ML_Infr_Lambda"!
```

---

### The Solution: Environment-Aware Detection

**File:** `src/path_config.py`

```python
def find_project_root(root_folder_name="CICD_ML_Infr_Lambda"):
    """
    Three-tier detection strategy:
    1. Lambda environment variables (highest priority)
    2. Search by folder name (local dev)
    3. Error with clear diagnostics
    """
    
    # Tier 1: Lambda detection via environment variables
    if os.environ.get('LAMBDA_TASK_ROOT'):
        # AWS sets this to '/var/task' in Lambda
        return Path(os.environ['LAMBDA_TASK_ROOT'])
    
    if os.environ.get('AWS_EXECUTION_ENV'):
        # Alternative Lambda indicator
        return Path('/var/task')
    
    # Tier 2: Local development - search by name
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if parent.name == root_folder_name:
            return parent
    
    # Tier 3: Failure with diagnostic info
    raise RuntimeError(
        f"Could not find project root '{root_folder_name}'. "
        f"Searched from: {current}"
    )
```

**Why this architecture?**

```
Environment Detection Priority:

1. Explicit signals (env vars)    ← Most reliable
   └─ Lambda sets LAMBDA_TASK_ROOT
   └─ Can't be spoofed/confused

2. Implicit signals (folder name)  ← Works locally
   └─ Searches parent directories
   └─ Matches exact string

3. Fail loudly                     ← No silent bugs
   └─ Shows where it searched
   └─ Clear error message
```

> **Critical Pattern:** Environment detection must handle **both known (Lambda) and unknown (future platforms)** execution contexts.

---

### Lambda's Filesystem Structure

```
/var/task/                        ← PROJECT_ROOT in Lambda
├── src/
│   ├── path_config.py           ← __file__ = /var/task/src/path_config.py
│   ├── model_loader.py
│   ├── inference_service.py
│   └── lambda_function.py
├── models/
│   └── model.pkl                ← Loaded from here
├── sklearn/                      ← Dependencies installed here
├── numpy/
└── ... (other deps)

/tmp/                             ← Only writable directory (512MB)
```

**Key insight:** Lambda extracts deployment package to `/var/task/`, which becomes read-only. The folder is **not** named after project.

---

## Phase 3: Three-Layer Architecture

### Separation of Concerns

**Design principle:** Each layer has one responsibility.

```
┌──────────────────────────────────────┐
│   lambda_function.py                 │  Layer 1: AWS Adapter
│   - Parse Lambda events              │  Thin, hard to test
│   - Format Lambda responses          │  
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│   inference_service.py               │  Layer 2: Business Logic  
│   - Validate input                   │  Pure Python, 100% testable
│   - Run inference                    │
│   - Format predictions               │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│   model_loader.py                    │  Layer 3: Resource Management
│   - Load model once (cold start)     │  Singleton pattern
│   - Cache in memory (warm starts)    │
└──────────────────────────────────────┘
```

---

### Layer 1: Lambda Handler

**File:** `src/lambda_function.py`

```python
def lambda_handler(event: Dict, context: Any) -> Dict:
    """
    Responsibility: Translate AWS events → business logic → AWS responses
    
    Supports two invocation patterns:
    1. Direct invoke: event = {"sepal_length": 5.1, ...}
    2. API Gateway: event = {"body": '{"sepal_length": 5.1, ...}'}
    """
    try:
        # Parse input based on source
        if 'body' in event:
            features = json.loads(event['body'])  # API Gateway
        else:
            features = event  # Direct invoke
        
        # Delegate to business logic
        result = predict(features)
        
        # Format AWS response
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(result)
        }
    
    except ValueError as e:
        # Client errors (bad input)
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid input', 'message': str(e)})
        }
    
    except Exception as e:
        # Server errors (unexpected)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal error', 'message': str(e)})
        }
```

**Why this pattern?**

- Handler has **zero business logic**
- Only knows about AWS-specific formats
- Business logic (predict) is reusable in FastAPI, Flask, etc.
- Easy to add API Gateway later without changing business code

---

### Layer 2: Business Logic

**File:** `src/inference_service.py`

```python
class InferenceService:
    """Pure business logic - no AWS dependencies"""
    
    EXPECTED_FEATURES = [
        'sepal_length', 'sepal_width',
        'petal_length', 'petal_width'
    ]
    
    def validate_input(self, features: Dict[str, float]) -> None:
        """
        Fail-fast validation.
        Better to catch errors here than in model.predict().
        """
        # Check for missing features
        missing = set(self.EXPECTED_FEATURES) - set(features.keys())
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Check for unexpected features
        extra = set(features.keys()) - set(self.EXPECTED_FEATURES)
        if extra:
            raise ValueError(f"Unexpected features: {extra}")
        
        # Type and range validation
        for key, value in features.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be numeric")
            if not (0 <= value <= 100):
                raise ValueError(f"{key} outside reasonable range")
    
    def predict(self, features: Dict[str, float]) -> Dict:
        """
        Core inference logic.
        
        Critical: Feature order must match training!
        """
        self.validate_input(features)
        
        # Model expects features in EXACT order they were trained
        feature_array = np.array([[
            features[name] for name in self.EXPECTED_FEATURES
        ]])
        
        # Get prediction
        prediction_idx = self.model.predict(feature_array)[0]
        probabilities = self.model.predict_proba(feature_array)[0]
        
        # Map index → class name
        class_name = self.metadata['target_names'][prediction_idx]
        confidence = float(probabilities[prediction_idx])
        
        return {
            'prediction': class_name,
            'confidence': confidence,
            'probabilities': probabilities.tolist(),
            'model_version': self.metadata['version']
        }
```

**Key insight:** Feature order preservation.

```python
# Why this matters:
model.fit(X_train)  # X_train columns: [sepal_length, sepal_width, ...]

# During prediction, ORDER MUST MATCH:
features = {
    'petal_width': 0.2,   # Dict order doesn't matter
    'sepal_length': 5.1,
    'petal_length': 1.4,
    'sepal_width': 3.5
}

# Wrong: use dict values directly
X = np.array([list(features.values())])  # ❌ Order is random!

# Right: explicit ordering
X = np.array([[features[name] for name in EXPECTED_FEATURES]])  # ✅
```

---

## Phase 4: The Singleton Pattern

### The Cold Start Problem

**Lambda execution model:**

```
Request 1 (Cold Start):
├── 1. Container created (~500ms)
├── 2. Python runtime initialized (~500ms)
├── 3. Code loaded (~200ms)
├── 4. Imports executed (~1000ms sklearn)
├── 5. model.pkl loaded (~100ms)        ← EXPENSIVE
└── 6. Handler executes (~10ms)
Total: ~2.3 seconds

Request 2-N (Warm):
├── Container already exists
├── Python already initialized
├── Imports already done
└── Handler executes (~10ms)             ← FAST
Total: ~10ms
```

**Problem:** Without caching, model loads every request.

```python
# ❌ BAD: Load on every invocation
def predict(features):
    model = pickle.load(open('model.pkl', 'rb'))  # 100ms each time!
    return model.predict(features)

# 1000 requests = 100 seconds wasted loading
```

---

### Singleton Implementation

**File:** `src/model_loader.py`

```python
class ModelLoader:
    """
    Singleton pattern ensures:
    1. Only one instance of ModelLoader exists
    2. Model loads once per container lifetime
    3. Subsequent calls return cached model
    """
    
    # Class-level storage (shared across all instances)
    _instance: Optional['ModelLoader'] = None
    _model: Optional[Any] = None
    _metadata: Optional[Dict] = None
    
    def __new__(cls):
        """
        __new__ controls instance creation.
        Called BEFORE __init__.
        """
        if cls._instance is None:
            # First time: create instance
            cls._instance = super().__new__(cls)
        # Always return the SAME instance
        return cls._instance
    
    def __init__(self):
        """
        __init__ called every time ModelLoader() is invoked,
        but we check if already initialized.
        """
        if self._model is not None:
            # Already loaded, skip
            return
        
        # Set paths (but don't load yet - lazy loading)
        self.model_path = MODELS_DIR / 'model.pkl'
        self.metadata_path = MODELS_DIR / 'model_metadata.json'
    
    def load_model(self) -> Any:
        """
        Lazy loading: only load when first requested.
        """
        if self._model is None:
            # First call: load from disk
            with open(self.model_path, 'rb') as f:
                self._model = pickle.load(f)
            print(f"Model loaded from: {self.model_path}")
        
        # Subsequent calls: return cached
        return self._model
```

**How singleton works:**

```python
# First invocation:
loader1 = ModelLoader()  # __new__ creates instance
model1 = loader1.load_model()  # Loads from disk

# Second invocation (even in same request):
loader2 = ModelLoader()  # __new__ returns SAME instance
model2 = loader2.load_model()  # Returns cached model

# Proof:
assert loader1 is loader2  # Same object in memory
assert model1 is model2    # Same model object
```

**Memory layout:**

```
Lambda Container Memory:

┌─────────────────────────────────────┐
│ Python Runtime                      │
│  ├── Imported modules               │
│  ├── ModelLoader._instance ───────┐ │
│  └── ModelLoader._model ─────────┐│ │
└──────────────────────────────────│││─┘
                                   │││
┌──────────────────────────────────│││─┐
│ Heap                             │││ │
│  ├── ModelLoader instance <──────┘│ │
│  └── RandomForest model <─────────┘ │
└─────────────────────────────────────┘
```

> **Pattern:** Singleton + lazy loading = optimal cold start performance.

---

## Phase 5: AWS SAM Deep Dive

### What is SAM?

**SAM = Serverless Application Model** — a framework on top of CloudFormation.

**Analogy:**

```
CloudFormation : Assembly Language
SAM            : High-level Language
```

**What SAM provides:**

- Shorthand for Lambda + API Gateway + other serverless resources
- Local testing (Docker-based)
- Automated packaging (dependencies, code)
- Deployment management

---

### The SAM Template

**File:** `template.yaml`

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31  # ← Enables SAM syntax

Resources:
  MLInferenceFunction:
    Type: AWS::Serverless::Function  # ← SAM shorthand
    Properties:
      CodeUri: .                     # Package current directory
      Handler: src.lambda_function.lambda_handler
      Runtime: python3.11
      MemorySize: 512
      Timeout: 30
      Architectures:
        - x86_64
      Environment:
        Variables:
          MODEL_VERSION: v1.0
          PYTHONPATH: /var/task       # ← Critical for imports
```

**What this generates in CloudFormation:**

```yaml
# SAM expands to ~100+ lines of CloudFormation:
Resources:
  MLInferenceFunction:
    Type: AWS::Lambda::Function
    Properties:
      Code:
        S3Bucket: !Ref DeploymentBucket
        S3Key: !Sub ${AWS::StackName}/code.zip
      # ... 50 more properties
  
  MLInferenceFunctionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        # ... complex IAM policy
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  
  MLInferenceFunctionLogGroup:
    Type: AWS::Logs::LogGroup
    # ... logging config
```

> **Pattern:** SAM abstracts complexity. You write 15 lines, AWS creates 6 resources.

---

### SAM Build Process

```bash
sam build
```

**What happens internally:**

```
Step 1: Parse template.yaml
├── Find CodeUri: .
├── Find Runtime: python3.11
└── Find Handler: src.lambda_function.lambda_handler

Step 2: Create build directory
├── Create: .aws-sam/build/MLInferenceFunction/
└── This will become the /var/task/ in Lambda

Step 3: Install dependencies
├── Read: requirements.txt
├── Download: scikit-learn==1.5.2
├── Download: numpy==1.26.4
└── Install to: .aws-sam/build/MLInferenceFunction/

Step 4: Copy source code
├── Copy: src/ → .aws-sam/build/MLInferenceFunction/src/
├── Copy: models/ → .aws-sam/build/MLInferenceFunction/models/
└── Result: Complete deployment package

Step 5: Generate deployment template
└── Create: .aws-sam/build/template.yaml (processed)
```

**Resulting structure:**

```
.aws-sam/build/MLInferenceFunction/
├── src/
│   ├── __init__.py
│   ├── path_config.py
│   ├── model_loader.py
│   ├── inference_service.py
│   └── lambda_function.py
├── models/
│   ├── model.pkl
│   └── model_metadata.json
├── sklearn/                    # Installed dependency
│   └── ... (many files)
├── numpy/                      # Installed dependency
│   └── ... (many files)
└── ... (other dependencies)

Total size: ~50MB (well under 250MB Lambda limit)
```

---

### SAM Deploy Process

```bash
sam deploy --resolve-s3
```

**Deployment flow:**

```
Step 1: Create/Find S3 Bucket
├── Check: Does deployment bucket exist?
├── No: Create aws-sam-cli-managed-default-samclisourcebucket-XXXXX
└── Yes: Use existing bucket

Step 2: Package Code
├── Zip: .aws-sam/build/MLInferenceFunction/
├── Upload to S3: s3://bucket/mini-ml-lambda/code.zip
└── Size: ~50MB compressed

Step 3: Generate CloudFormation Template
├── Replace CodeUri with S3 location
├── Add IAM roles, policies
├── Add CloudWatch log groups
└── Create change set

Step 4: Execute CloudFormation Stack
├── Create: Lambda function
├── Create: IAM execution role
├── Create: CloudWatch log group
├── Set: Environment variables
└── Configure: Memory, timeout, runtime

Step 5: Output Results
└── Print: Function ARN, name, etc.
```

**CloudFormation change set example:**

```json
{
  "Changes": [
    {
      "Type": "Resource",
      "Action": "Add",
      "ResourceType": "AWS::Lambda::Function",
      "LogicalResourceId": "MLInferenceFunction",
      "Details": {
        "FunctionName": "mini-ml-lambda-stack-MLInferenceFunction-ABC123",
        "Runtime": "python3.11",
        "MemorySize": 512,
        "CodeUri": "s3://bucket/code.zip"
      }
    }
  ]
}
```

---

## Phase 6: Lambda Environment Mysteries

### How Lambda Executes Code

**Invocation lifecycle:**

```
1. API Gateway/Direct Invoke
   └─> Lambda service receives event

2. Container Selection
   ├─> Cold start: Create new container
   └─> Warm start: Reuse existing container

3. Code Extraction (Cold Start Only)
   ├─> Download code.zip from S3
   ├─> Extract to /var/task/
   └─> Set permissions (read-only)

4. Runtime Initialization (Cold Start Only)
   ├─> Start Python 3.11 interpreter
   ├─> Set environment variables
   │   ├── LAMBDA_TASK_ROOT=/var/task
   │   ├── AWS_EXECUTION_ENV=AWS_Lambda_python3.11
   │   └── MODEL_VERSION=v1.0 (from template)
   └─> Execute module imports

5. Handler Invocation (Every Request)
   ├─> Call: lambda_function.lambda_handler(event, context)
   ├─> Capture: stdout, stderr → CloudWatch Logs
   └─> Return: response or error

6. Container Lifecycle
   ├─> Keep warm: 5-15 minutes
   └─> Freeze after idle timeout
```

---

### Environment Variables in Lambda

**How authentication works:**

```python
# workflow sets:
AWS_ACCESS_KEY_ID: AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY: wJalrXUtn...
AWS_REGION: us-east-1

# SAM CLI internally does:
import boto3

# These env vars are automatically read by boto3
lambda_client = boto3.client('lambda')  

# boto3 makes API call:
# POST https://lambda.us-east-1.amazonaws.com/
# Headers: {
#   'Authorization': 'AWS4-HMAC-SHA256 Credential=AKIA.../us-east-1/lambda/...',
#   'X-Amz-Date': '20251201T123456Z'
# }

# AWS backend:
# 1. Decrypts access key → finds Account ID: 123456789012
# 2. Verifies secret key matches
# 3. Checks IAM permissions
# 4. Allows/denies operation
```

**Key insight:** Access keys encode account information.

```
Access Key Structure:
AKIA IOSFODNN 7EXAMPLE
│    │        └─ Random identifier
│    └─ Key type indicator
└─ Always "AKIA" for access keys

AWS Internal Mapping:
AKIAIOSFODNN7EXAMPLE → {
  "AccountId": "123456789012",
  "UserId": "AIDAI23HXS4LK7EXAMPLE",
  "UserName": "github-actions-deployer"
}
```

> **Pattern:** Credentials carry identity. No need to specify account ID explicitly.

---

### Lambda-Specific Imports

**Why `PYTHONPATH=/var/task` matters:**

```python
# code:
from src.inference_service import predict

# Python import resolution:
import sys
# sys.path = [
#   '/var/task',          ← Set by PYTHONPATH env var
#   '/opt/python',
#   '/var/runtime',
#   ...
# ]

# Python searches:
# 1. /var/task/src/inference_service.py ✅ Found!
# 2. Loads module
# 3. Executes imports inside that file
```

**Without PYTHONPATH:**

```python
# sys.path = [
#   '/var/runtime',       ← Lambda's built-in libs only
#   ...
# ]

from src.inference_service import predict
# ModuleNotFoundError: No module named 'src'
```

---

## Phase 7: GitHub Actions Pipeline

### CI/CD Workflow Architecture

**File:** `.github/workflows/deploy-ml-lambda.yml`

```yaml
# Trigger conditions
on:
  push:
    branches: [main]
    paths: ['CICD_ML_Infr_Lambda/**']  # Only this folder
  workflow_dispatch:  # Manual trigger option

# Two-stage pipeline with dependency
jobs:
  test:
    # Stage 1: Quality gate
  
  deploy:
    needs: test  # ← Depends on test passing
    # Stage 2: Deployment
```

---

### Test Job Deep Dive

```yaml
test:
  runs-on: ubuntu-latest
  defaults:
    run:
      working-directory: CICD_ML_Infr_Lambda  # All commands run here
  
  steps:
    - name: Checkout code
      uses: actions/checkout@v4
      # Clones: mjsushanth/mlops-labs-portfolio
      
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
      # Installs Python 3.11.x on Ubuntu runner
      
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
      # Installs: pytest, coverage, sklearn, numpy
      
    - name: Run tests with coverage
      run: |
        pytest tests/ -v \
          --cov=src \
          --cov-report=term-missing \
          --cov-fail-under=80
      # Runs 25 tests, requires ≥80% coverage
      # If fails → pipeline stops here ❌
```

**Why `--cov-fail-under=80`?**

```bash
# Scenario 1: Coverage drops
# Developer removes tests accidentally
pytest tests/ --cov=src --cov-fail-under=80
# Coverage: 75%
# ❌ FAILED: Coverage below 80%
# Deploy job never runs → bad code doesn't reach production

# Scenario 2: Coverage maintained
pytest tests/ --cov=src --cov-fail-under=80
# Coverage: 97%
# ✅ PASSED
# Deploy job runs → good code reaches production
```

> **Pattern:** Automated quality gates prevent regressions.

---

### Deploy Job Deep Dive

```yaml
deploy:
  needs: test  # Only runs if test job succeeds
  runs-on: ubuntu-latest
  defaults:
    run:
      working-directory: CICD_ML_Infr_Lambda
  
  steps:
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
      # Sets environment variables for AWS SDK
      
    - name: SAM Deploy
      run: |
        sam deploy \
          --stack-name mini-ml-lambda-stack \
          --s3-prefix mini-ml-lambda \
          --region us-east-1 \
          --capabilities CAPABILITY_IAM \
          --resolve-s3 \
          --no-confirm-changeset \
          --no-fail-on-empty-changeset
```

**Critical flags explained:**

```bash
--resolve-s3
# Creates/finds S3 bucket automatically
# Without this: ERROR: S3 Bucket not specified

--capabilities CAPABILITY_IAM
# Allows SAM to create IAM roles
# Without this: ERROR: Insufficient permissions

--no-confirm-changeset
# Skip manual approval (automated pipeline)
# With this: Deploys immediately after build

--no-fail-on-empty-changeset
# Don't error if nothing changed
# Useful for re-running workflows
```

---

### Secrets Management

**GitHub Secrets storage:**

```
Repository Settings → Secrets and Variables → Actions

Secrets (encrypted at rest):
├── AWS_ACCESS_KEY_ID: AKIAIOSFODNN7EXAMPLE
├── AWS_SECRET_ACCESS_KEY: wJalrXUtn... (never shown)
└── AWS_REGION: us-east-1

Workflow access:
${{ secrets.AWS_ACCESS_KEY_ID }}  # Masked in logs: ***
```

**Security model:**

```yaml
# In workflow logs:
Run: aws sts get-caller-identity
{
  "Account": "123456789012",
  "UserId": "AIDAI***",
  "Arn": "arn:aws:iam::123456789012:user/github-actions-deployer"
}

# Secrets never appear in logs:
Configure AWS credentials
  with:
    aws-access-key-id: ***          ← Automatically masked
    aws-secret-access-key: ***
```

---

## Key Learnings & Patterns

### 1. Serverless Pricing Model

**Lambda cost formula:**

```
Cost = (Requests × $0.20/million) + (GB-seconds × $0.0000166667)

configuration:
├── Memory: 512 MB = 0.5 GB
├── Avg duration: 50ms (warm) = 0.05 seconds
└── GB-seconds per request: 0.5 × 0.05 = 0.025

10,000 requests/month:
├── Request cost: 10,000 × $0.0000002 = $0.002
├── Compute cost: 10,000 × 0.025 × $0.0000166667 = $0.0042
└── Total: $0.0062 ≈ $0.01/month
```

**Free tier:**

```
Always free (doesn't expire):
├── 1 million requests/month
└── 400,000 GB-seconds/month

usage (10K requests):
├── Requests: 1% of free tier
└── GB-seconds: 6.25% of free tier

Conclusion: $0.00/month ✅
```

---

### 2. Cold Start Optimization

**Optimization strategy:**

```
1. Minimize package size
   ├── Only include necessary dependencies
   ├── No dev dependencies (pytest, etc.)
   └── Result: 50MB vs 200MB

2. Lazy loading
   ├── Don't import heavy modules at file level
   ├── Import inside functions when needed
   └── Result: Faster import time

3. Singleton pattern
   ├── Load model once per container
   ├── Cache in memory
   └── Result: 100ms → 0ms on warm starts

4. Provisioned concurrency (optional, costs money)
   ├── Keep N containers always warm
   └── Result: 0ms cold starts
```

**Trade-offs:**

```
Approach             | Cold Start | Warm Start | Cost
---------------------|------------|------------|--------
No optimization      | 5000ms     | 2500ms     | Low
Singleton only       | 2500ms     | 10ms       | Low
Provisioned (1 unit) | 10ms       | 10ms       | $13/mo
```

---

### 3. Monorepo CI/CD Pattern

**Path-based triggers:**

```yaml
on:
  push:
    paths:
      - 'CICD_ML_Infr_Lambda/**'  # Only this project
```

**Why this matters:**

```
Repository: mlops-labs-portfolio/
├── API_Labs/                   ← Change here
├── Docker_GCP_FlaskMLApp/      ← Or here
└── CICD_ML_Infr_Lambda/        ← Doesn't trigger workflow

Repository: mlops-labs-portfolio/
├── CICD_ML_Infr_Lambda/
│   └── src/
│       └── inference_service.py  ← Change here
                                  → Triggers workflow ✅
```

**Benefits:**

- Multiple projects, separate workflows
- No unnecessary CI/CD runs
- Each project independently deployable
- Cost savings (fewer workflow minutes)

---

### 4. Testing Strategy

**Test pyramid for ML systems:**

```
                    ▲
                   ╱ ╲
                  ╱   ╲
                 ╱ E2E ╲         ← 0 tests (expensive, slow)
                ╱───────╲
               ╱         ╲
              ╱Integration╲      ← 0 tests (would need AWS)
             ╱─────────────╲
            ╱               ╲
           ╱   Unit Tests    ╲   ← 25 tests (fast, thorough)
          ╱___________________╲
```

**What we tested:**

```python
# Model Loader Tests (8)
├── Singleton behavior
├── Caching
├── Path resolution
└── Metadata loading

# Inference Service Tests (13)
├── Valid predictions
├── Input validation
│   ├── Missing features
│   ├── Extra features
│   ├── Invalid types
│   └── Out-of-range values
├── Probability validation
└── Response format

# Lambda Handler Tests (4)
├── Direct invocation
├── API Gateway invocation
├── Error handling (400)
└── Error handling (500)
```

**Why 97% coverage is enough:**

```python
# Covered: Core logic
def predict(features):          # ✅ 100% coverage
    validate_input(features)    # ✅ All branches tested
    result = model.predict()    # ✅ Tested
    return format_response()    # ✅ Tested

# Not covered: Edge cases
if os.environ.get('LAMBDA_TASK_ROOT'):  # ❌ Can't test Lambda env locally
    return Path('/var/task')

# Acceptable because:
# 1. Lambda environment is AWS's responsibility
# 2. SAM local invoke tests this path
# 3. Production deployment validates it
```

---

### 5. Infrastructure as Code Benefits

**Version control for infrastructure:**

```yaml
# template.yaml is code
Resources:
  MLInferenceFunction:
    Type: AWS::Serverless::Function
    Properties:
      MemorySize: 512  # ← Change this, commit, push

# Git history shows infrastructure changes:
commit a1b2c3d
  Changed MemorySize: 256 → 512
  Reason: Cold starts >5s, need more CPU

commit d4e5f6g
  Added Environment variable: LOG_LEVEL
  Reason: Enable debug logging
```

**Benefits:**

- Infrastructure changes reviewed like code
- Rollback via `git revert`
- Same deployment in dev/staging/prod
- No "it works on my machine"

---

## Advanced Concepts

### Lambda Execution Context Reuse

**How Lambda reuses containers:**

```python
# Module level (executed once per container)
print("Container initialized")
model_loader = ModelLoader()  # Singleton created

def lambda_handler(event, context):
    # Executed every invocation
    print(f"Request ID: {context.request_id}")
    model = model_loader.get_model()  # Returns cached
    return predict(model, event)

# Request lifecycle:
Request 1 (Cold):
  Container initialized    ← Printed once
  Request ID: abc123       ← Printed
  
Request 2 (Warm):
  Request ID: def456       ← Printed (no "Container initialized")
  
Request 3 (Warm):
  Request ID: ghi789       ← Printed
```

**Memory between invocations:**

```python
# Globals persist between warm invocations
_request_count = 0

def lambda_handler(event, context):
    global _request_count
    _request_count += 1
    print(f"This container has served {_request_count} requests")

# Output:
Request 1: This container has served 1 requests
Request 2: This container has served 2 requests
Request 3: This container has served 3 requests
```

> **Warning:** Don't rely on this for state. Containers can be terminated anytime.

---

### SAM Local Testing Deep Dive

**How `sam local invoke` works:**

```bash
sam local invoke MLInferenceFunction --event events/test_event.json
```

**Under the hood:**

```
1. SAM reads template.yaml
   └─> Find: MLInferenceFunction definition

2. SAM checks Docker
   └─> Is Lambda Python 3.11 image available?

3. SAM pulls image (first time only)
   └─> docker pull public.ecr.aws/lambda/python:3.11

4. SAM starts container
   └─> docker run -v .aws-sam/build/:/var/task ...

5. SAM mounts code
   └─> /var/task/ inside container = .aws-sam/build/MLInferenceFunction/

6. SAM sets environment variables
   ├─> LAMBDA_TASK_ROOT=/var/task
   ├─> AWS_EXECUTION_ENV=AWS_Lambda_python3.11
   └─> MODEL_VERSION=v1.0

7. SAM invokes handler
   └─> Python lambda_function.lambda_handler(event, context)

8. SAM captures output
   └─> Return response + logs
```

**Why this is valuable:**

```
Local testing catches:
├── Import errors
├── Path resolution issues  
├── Missing dependencies
├── Environment variable bugs
└── Logic errors

Before deploying to AWS:
├── No AWS charges for testing
├── Faster iteration (no upload to S3)
└── Works offline
```

---

## Troubleshooting Guide

### Common Issues & Solutions

**Issue 1: ModuleNotFoundError in Lambda**

```python
# Error:
ModuleNotFoundError: No module named 'src'

# Cause:
PYTHONPATH not set in Lambda environment

# Solution:
# template.yaml
Environment:
  Variables:
    PYTHONPATH: /var/task  # ← Add this
```

---

**Issue 2: Model file not found**

```python
# Error:
FileNotFoundError: model.pkl not found

# Cause:
Path resolution assumes wrong root directory

# Solution:
# Use environment-aware detection
if os.environ.get('LAMBDA_TASK_ROOT'):
    ROOT = Path(os.environ['LAMBDA_TASK_ROOT'])
```

---

**Issue 3: Lambda timeout**

```python
# Error:
Task timed out after 3.00 seconds

# Cause:
Model loading on every invocation

# Solution:
# Use singleton pattern
_model_cache = None
def get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = pickle.load(...)
    return _model_cache
```

---

**Issue 4: GitHub Actions workflow not triggering**

```yaml
# Cause:
Workflow file not in .github/workflows/ at repo root

# Wrong:
CICD_ML_Infr_Lambda/.github/workflows/deploy.yml  ❌

# Right:
.github/workflows/deploy-ml-lambda.yml  ✅
```

---

## Production Considerations

### Monitoring & Observability

**CloudWatch metrics to monitor:**

```
1. Invocations
   └─> How many requests?

2. Errors
   └─> How many failed?

3. Duration
   ├─> Cold start: Should be <5s
   └─> Warm start: Should be <100ms

4. Throttles
   └─> Hitting concurrency limits?

5. Iterator age (for event sources)
   └─> N/A for this use case
```

**Setting up alarms:**

```yaml
# In template.yaml
ErrorAlarm:
  Type: AWS::CloudWatch::Alarm
  Properties:
    MetricName: Errors
    Namespace: AWS/Lambda
    Statistic: Sum
    Period: 300
    EvaluationPeriods: 1
    Threshold: 10
    ComparisonOperator: GreaterThanThreshold
    AlarmActions:
      - !Ref SNSTopic  # Send notification
```

---

### Security Best Practices

**IAM least privilege:**

```yaml
# Current: Full access policies (for learning)
github-actions-deployer:
  - AWSLambdaFullAccess
  - IAMFullAccess
  - AmazonS3FullAccess
  - AWSCloudFormationFullAccess

# Production: Minimal required permissions
github-actions-deployer:
  PolicyDocument:
    Statement:
      - Effect: Allow
        Action:
          - lambda:UpdateFunctionCode
          - lambda:UpdateFunctionConfiguration
        Resource: !GetAtt MLInferenceFunction.Arn
      
      - Effect: Allow
        Action:
          - s3:PutObject
        Resource: !Sub ${DeploymentBucket.Arn}/*
```

---

### Cost Optimization

**Strategies:**

```
1. Right-size memory
   ├── Start: 512 MB
   ├── Monitor: Actual usage (198 MB in our case)
   └── Reduce: To 256 MB → save 50% on compute cost

2. Reduce package size
   ├── Remove: Unused dependencies
   ├── Use: Lambda layers for common dependencies
   └── Result: Faster cold starts

3. Batch processing
   ├── Instead: 1000 individual invocations
   ├── Use: 1 invocation processing batch of 1000
   └── Result: 1000× fewer requests billed

4. Reserved concurrency
   ├── If: Unpredictable spikes
   ├── Set: Reserved concurrency = 0
   └── Result: Prevent surprise bills (throttles instead)
```

---

## Conclusion

### What We Achieved

**Technical accomplishments:**

- Production-ready ML inference API
- Full CI/CD automation
- 97% test coverage
- Clean architecture (SOLID principles)
- Cost-optimized ($0.00/month)
- Deployed and operational

**Patterns demonstrated:**

- Singleton for resource management
- Environment-aware path resolution
- Three-layer architecture
- Infrastructure as Code
- Automated testing gates
- Secure secret management

**Time efficiency:**

- 75 minutes from zero to production
- ~45 prompts/iterations
- 15 files created/modified
- 800+ lines of code
- 6 AWS resources created

---

### Next Steps (Optional Enhancements)

**1. Add API Gateway**

```yaml
Events:
  ApiEvent:
    Type: Api
    Properties:
      Path: /predict
      Method: post
```

**2. Model versioning**

```
models/
├── v1.0/
│   └── model.pkl
└── v2.0/
    └── model.pkl

# Switch via environment variable
MODEL_VERSION=v2.0
```

**3. A/B testing**

```python
# Route 10% traffic to new model
if random.random() < 0.1:
    model = load_model('v2.0')
else:
    model = load_model('v1.0')
```

**4. CloudWatch dashboard**

```
Metrics:
├── Invocations/minute
├── Error rate
├── P50/P95/P99 latency
└── Cost/request
```

---

### References & Resources

**AWS Documentation:**

- [Lambda Python Runtime](https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html)
- [SAM CLI Reference](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-command-reference.html)
- [CloudFormation Resource Types](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-template-resource-type-ref.html)

**Design Patterns:**

- Singleton: Gang of Four Design Patterns
- Three-layer architecture: Clean Architecture (Robert C. Martin)
- Infrastructure as Code: The Phoenix Project

**Project Repository:**

- https://github.com/mjsushanth/mlops-labs-portfolio/tree/main/CICD_ML_Infr_Lambda

---
