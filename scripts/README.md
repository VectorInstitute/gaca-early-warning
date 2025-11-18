# Deployment Scripts

This directory contains scripts for deploying the GACA Early Warning System to cloud platforms.

## Cloud Run Deployment

### Prerequisites

1. **Google Cloud SDK**: Install and configure the gcloud CLI
   ```bash
   # Install gcloud CLI (if not already installed)
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL

   # Initialize and authenticate
   gcloud init
   gcloud auth login
   ```

2. **Project Access**: Ensure you have the necessary permissions in the `coderd` GCP project:
   - Cloud Run Admin
   - Cloud Build Editor
   - Service Account User
   - Artifact Registry Administrator

3. **Docker**: Not required! The deployment uses Cloud Build which handles containerization.

### Quick Start

Deploy both backend and dashboard:

```bash
./scripts/deploy_to_cloud_run.sh --allow-unauthenticated
```

### Deployment Options

#### Deploy Both Services (Default)
```bash
./scripts/deploy_to_cloud_run.sh \
  --project coderd \
  --region us-central1 \
  --allow-unauthenticated
```

#### Deploy Backend Only
```bash
./scripts/deploy_to_cloud_run.sh \
  --backend-only \
  --project coderd \
  --region us-central1
```

#### Deploy Dashboard Only
```bash
./scripts/deploy_to_cloud_run.sh \
  --app-only \
  --project coderd \
  --region us-central1
```

#### Dry Run (Preview Commands)
```bash
./scripts/deploy_to_cloud_run.sh --dry-run
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--project PROJECT_ID` | `coderd` | GCP project ID |
| `--region REGION` | `us-central1` | Cloud Run region |
| `--backend-service NAME` | `gaca-ews-backend` | Backend service name |
| `--app-service NAME` | `gaca-ews-dashboard` | Dashboard service name |
| `--allow-unauthenticated` | `false` | Allow public access |
| `--backend-only` | - | Deploy only backend |
| `--app-only` | - | Deploy only dashboard |
| `--dry-run` | - | Preview without executing |

### Resource Configuration

#### Backend (FastAPI)
- **Memory**: 2 GiB
- **CPU**: 2 vCPU
- **Timeout**: 300s
- **Concurrency**: 10 requests per instance
- **Min Instances**: 0 (scale to zero)
- **Max Instances**: 5
- **Port**: 8080

#### Dashboard (Next.js)
- **Memory**: 1 GiB
- **CPU**: 1 vCPU
- **Timeout**: 60s
- **Concurrency**: 80 requests per instance
- **Min Instances**: 0 (scale to zero)
- **Max Instances**: 10
- **Port**: 3000

### Architecture

```
┌──────────────────┐
│   User Browser   │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────┐
│  GACA EWS Dashboard          │
│  (Next.js on Cloud Run)      │
│  - Interactive UI            │
│  - Real-time updates         │
│  - Map visualizations        │
└────────┬─────────────────────┘
         │ HTTPS
         ▼
┌──────────────────────────────┐
│  GACA EWS Backend            │
│  (FastAPI on Cloud Run)      │
│  - Model inference           │
│  - NOAA data fetching        │
│  - WebSocket support         │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  NOAA URMA S3 Bucket         │
│  (noaa-urma-pds)             │
│  - Meteorological data       │
└──────────────────────────────┘
```

### Deployment Flow

1. **Backend Deployment**:
   - Cloud Build packages the FastAPI app with all model artifacts
   - Creates container image in Artifact Registry
   - Deploys to Cloud Run with 2GB memory for model inference
   - Exposes endpoints: `/health`, `/model/info`, `/predict`, `/ws/predict`

2. **Dashboard Deployment**:
   - Retrieves backend URL from deployed backend service
   - Cloud Build packages Next.js app with backend URL as env var
   - Creates optimized production build with standalone output
   - Deploys to Cloud Run with backend connection configured

3. **CORS Configuration**:
   - Backend automatically allows Cloud Run URLs via regex: `https://.*\.run\.app`
   - Supports custom origins via `ALLOWED_ORIGINS` environment variable

### Post-Deployment

After successful deployment, the script will output:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Deployment Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Backend Service:
  URL: https://gaca-ews-backend-XXXXXX.us-central1.run.app
  Endpoints:
    - GET  https://gaca-ews-backend-XXXXXX.us-central1.run.app/
    - GET  https://gaca-ews-backend-XXXXXX.us-central1.run.app/health
    - GET  https://gaca-ews-backend-XXXXXX.us-central1.run.app/model/info
    - POST https://gaca-ews-backend-XXXXXX.us-central1.run.app/predict
    - WS   https://gaca-ews-backend-XXXXXX.us-central1.run.app/ws/predict

Dashboard App:
  URL: https://gaca-ews-dashboard-XXXXXX.us-central1.run.app

Next Steps:
  1. Test the backend health endpoint:
     curl https://gaca-ews-backend-XXXXXX.us-central1.run.app/health

  2. Test the dashboard:
     Open https://gaca-ews-dashboard-XXXXXX.us-central1.run.app in your browser

  3. Update backend CORS if needed (see backend/app/main.py)
```

### Testing the Deployment

#### Test Backend Health
```bash
curl https://gaca-ews-backend-XXXXXX.us-central1.run.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Test Model Info
```bash
curl https://gaca-ews-backend-XXXXXX.us-central1.run.app/model/info
```

#### Test Inference (REST API)
```bash
curl -X POST https://gaca-ews-backend-XXXXXX.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"num_hours": 24}'
```

### Updating the Deployment

To update an existing deployment, simply run the deployment script again:

```bash
./scripts/deploy_to_cloud_run.sh --allow-unauthenticated
```

Cloud Run will:
- Build a new container image
- Gradually shift traffic to the new revision
- Keep previous revisions for rollback

### Rollback

If you need to rollback to a previous version:

```bash
# List revisions
gcloud run revisions list \
  --service=gaca-ews-backend \
  --region=us-central1 \
  --project=coderd

# Route traffic to specific revision
gcloud run services update-traffic gaca-ews-backend \
  --to-revisions=REVISION_NAME=100 \
  --region=us-central1 \
  --project=coderd
```

### Monitoring and Logs

#### View Logs
```bash
# Backend logs
gcloud run services logs read gaca-ews-backend \
  --region=us-central1 \
  --project=coderd \
  --limit=50

# Dashboard logs
gcloud run services logs read gaca-ews-dashboard \
  --region=us-central1 \
  --project=coderd \
  --limit=50
```

#### View Metrics
Visit the Cloud Run console:
- Backend: https://console.cloud.google.com/run/detail/us-central1/gaca-ews-backend
- Dashboard: https://console.cloud.google.com/run/detail/us-central1/gaca-ews-dashboard

### Troubleshooting

#### Build Failures

**Issue**: Cloud Build fails with "out of memory"
**Solution**: The backend uses 2GB memory. If builds still fail, check the Dockerfile for optimization opportunities.

**Issue**: Missing dependencies in container
**Solution**: Ensure all model artifacts are present in `data/` and `src/gaca_ews/model/`

#### Runtime Errors

**Issue**: Backend returns 503 "Model not loaded"
**Solution**: Check that all model files are included in the Docker image. View logs:
```bash
gcloud run services logs read gaca-ews-backend --region=us-central1 --project=coderd
```

**Issue**: Dashboard can't connect to backend
**Solution**: Verify backend URL is correctly set:
```bash
gcloud run services describe gaca-ews-dashboard \
  --region=us-central1 \
  --project=coderd \
  --format='value(spec.template.spec.containers[0].env)'
```

#### CORS Issues

**Issue**: Browser console shows CORS errors
**Solution**: The backend now uses regex to allow all `*.run.app` domains. If you need custom domains, set the `ALLOWED_ORIGINS` environment variable:
```bash
gcloud run services update gaca-ews-backend \
  --update-env-vars=ALLOWED_ORIGINS="https://custom-domain.com,https://another-domain.com" \
  --region=us-central1 \
  --project=coderd
```

### Cost Optimization

Cloud Run charges for:
- **Compute**: CPU and memory usage during request handling
- **Requests**: Number of requests
- **Networking**: Egress traffic

**Tips**:
- The deployment uses `min-instances=0` to scale to zero when idle
- Backend uses 2GB memory only during active inference
- First 2 million requests per month are free
- Consider setting budget alerts in GCP console

### Security

#### Authentication (Recommended for Production)

Remove `--allow-unauthenticated` flag for production:

```bash
./scripts/deploy_to_cloud_run.sh --project coderd
```

Then grant access to specific users:

```bash
# Grant access to backend
gcloud run services add-iam-policy-binding gaca-ews-backend \
  --region=us-central1 \
  --project=coderd \
  --member='user:email@example.com' \
  --role='roles/run.invoker'

# Grant access to dashboard
gcloud run services add-iam-policy-binding gaca-ews-dashboard \
  --region=us-central1 \
  --project=coderd \
  --member='user:email@example.com' \
  --role='roles/run.invoker'
```

### Environment Variables

#### Backend
- `ALLOWED_ORIGINS`: Comma-separated list of additional CORS origins

#### Dashboard
- `NEXT_PUBLIC_API_URL`: Backend API URL (automatically set during deployment)

### Support

For issues or questions:
- Open an issue in the GitHub repository
- Contact Vector AI Engineering team
- Check Cloud Run documentation: https://cloud.google.com/run/docs
