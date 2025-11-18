#!/bin/bash
#
# Deploy GACA Early Warning System (Backend + Dashboard) to Cloud Run
#
# This script builds and deploys both the FastAPI backend and Next.js dashboard
# to Google Cloud Run, ensuring proper connectivity between services.
#
# Usage:
#   ./scripts/deploy_to_cloud_run.sh [OPTIONS]
#
# Options:
#   --project PROJECT_ID           GCP project ID (default: coderd)
#   --region REGION               Cloud Run region (default: us-central1)
#   --backend-service NAME        Backend service name (default: gaca-ews-backend)
#   --app-service NAME            App service name (default: gaca-ews-dashboard)
#   --allow-unauthenticated       Allow unauthenticated requests (default: false)
#   --backend-only                Deploy only the backend service
#   --app-only                    Deploy only the app service
#   --dry-run                     Show commands without executing
#

set -euo pipefail

# Default configuration
PROJECT_ID="${GCP_PROJECT:-coderd}"
REGION="us-central1"
BACKEND_SERVICE_NAME="gaca-ews-backend"
APP_SERVICE_NAME="gaca-ews-dashboard"
ALLOW_UNAUTHENTICATED="false"
DEPLOY_BACKEND="true"
DEPLOY_APP="true"
DRY_RUN="false"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --project)
      PROJECT_ID="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --backend-service)
      BACKEND_SERVICE_NAME="$2"
      shift 2
      ;;
    --app-service)
      APP_SERVICE_NAME="$2"
      shift 2
      ;;
    --allow-unauthenticated)
      ALLOW_UNAUTHENTICATED="true"
      shift
      ;;
    --backend-only)
      DEPLOY_APP="false"
      shift
      ;;
    --app-only)
      DEPLOY_BACKEND="false"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_DIR="${PROJECT_ROOT}/backend"
APP_DIR="${PROJECT_ROOT}/app"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   GACA Early Warning System - Cloud Run Deployment${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo "  Project ID:          ${PROJECT_ID}"
echo "  Region:              ${REGION}"
echo "  Backend Service:     ${BACKEND_SERVICE_NAME}"
echo "  App Service:         ${APP_SERVICE_NAME}"
echo "  Project Root:        ${PROJECT_ROOT}"
echo "  Allow Unauth:        ${ALLOW_UNAUTHENTICATED}"
echo "  Deploy Backend:      ${DEPLOY_BACKEND}"
echo "  Deploy App:          ${DEPLOY_APP}"
echo "  Dry Run:             ${DRY_RUN}"
echo ""

# Function to execute or print commands
run_cmd() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo -e "${YELLOW}[DRY RUN]${NC} $*"
  else
    echo -e "${GREEN}▶${NC} $*"
    "$@"
  fi
}

# Verify required directories exist
if [[ "${DEPLOY_BACKEND}" == "true" ]] && [[ ! -d "${BACKEND_DIR}" ]]; then
  echo -e "${RED}✗ Backend directory not found: ${BACKEND_DIR}${NC}"
  exit 1
fi

if [[ "${DEPLOY_APP}" == "true" ]] && [[ ! -d "${APP_DIR}" ]]; then
  echo -e "${RED}✗ App directory not found: ${APP_DIR}${NC}"
  exit 1
fi

# Check required files
if [[ "${DEPLOY_BACKEND}" == "true" ]]; then
  BACKEND_REQUIRED_FILES=("backend/Dockerfile" "backend/app/main.py" "pyproject.toml")
  for file in "${BACKEND_REQUIRED_FILES[@]}"; do
    if [[ ! -f "${PROJECT_ROOT}/${file}" ]]; then
      echo -e "${RED}✗ Required file not found: ${file}${NC}"
      exit 1
    fi
  done
  echo -e "${GREEN}✓${NC} Backend files verified"
fi

if [[ "${DEPLOY_APP}" == "true" ]]; then
  APP_REQUIRED_FILES=("app/Dockerfile" "app/package.json" "app/next.config.ts")
  for file in "${APP_REQUIRED_FILES[@]}"; do
    if [[ ! -f "${PROJECT_ROOT}/${file}" ]]; then
      echo -e "${RED}✗ Required file not found: ${file}${NC}"
      exit 1
    fi
  done
  echo -e "${GREEN}✓${NC} App files verified"
fi
echo ""

# Step 1: Set GCP project
echo -e "${BLUE}Step 1: Configure GCP Project${NC}"
run_cmd gcloud config set project "${PROJECT_ID}"
echo ""

# Step 2: Enable required APIs
echo -e "${BLUE}Step 2: Enable Required APIs${NC}"
REQUIRED_APIS=(
  "run.googleapis.com"
  "cloudbuild.googleapis.com"
  "artifactregistry.googleapis.com"
)

for api in "${REQUIRED_APIS[@]}"; do
  echo -e "${GREEN}▶${NC} Enabling ${api}..."
  run_cmd gcloud services enable "${api}" --project="${PROJECT_ID}"
done
echo ""

# Step 3: Deploy Backend
BACKEND_URL=""
if [[ "${DEPLOY_BACKEND}" == "true" ]]; then
  echo -e "${BLUE}Step 3: Deploy Backend Service${NC}"
  echo -e "${GREEN}▶${NC} Building and deploying backend container..."

  # Copy backend Dockerfile to root temporarily (gcloud run deploy --source requires Dockerfile in source root)
  echo -e "${CYAN}Copying backend Dockerfile to project root...${NC}"
  if [[ "${DRY_RUN}" == "false" ]]; then
    cp "${BACKEND_DIR}/Dockerfile" "${PROJECT_ROOT}/Dockerfile"
  fi

  BACKEND_DEPLOY_CMD=(
    gcloud run deploy "${BACKEND_SERVICE_NAME}"
    --source="${PROJECT_ROOT}"
    --platform=managed
    --region="${REGION}"
    --project="${PROJECT_ID}"
    --memory=2Gi
    --cpu=2
    --timeout=300s
    --max-instances=5
    --min-instances=0
    --concurrency=10
    --port=8080
  )

  if [[ "${ALLOW_UNAUTHENTICATED}" == "true" ]]; then
    BACKEND_DEPLOY_CMD+=(--allow-unauthenticated)
    echo -e "${YELLOW}Note: Backend allows unauthenticated access${NC}"
  else
    BACKEND_DEPLOY_CMD+=(--no-allow-unauthenticated)
  fi

  run_cmd "${BACKEND_DEPLOY_CMD[@]}"

  # Clean up temporary Dockerfile
  if [[ "${DRY_RUN}" == "false" ]]; then
    echo -e "${CYAN}Cleaning up temporary Dockerfile...${NC}"
    rm -f "${PROJECT_ROOT}/Dockerfile"
  fi
  echo ""

  # Get backend URL
  if [[ "${DRY_RUN}" == "false" ]]; then
    echo -e "${CYAN}Retrieving backend service URL...${NC}"
    BACKEND_URL=$(gcloud run services describe "${BACKEND_SERVICE_NAME}" \
      --platform=managed \
      --region="${REGION}" \
      --project="${PROJECT_ID}" \
      --format='value(status.url)')

    echo -e "${GREEN}✓${NC} Backend deployed successfully!"
    echo -e "${GREEN}Backend URL:${NC} ${BACKEND_URL}"
    echo ""
  else
    BACKEND_URL="https://${BACKEND_SERVICE_NAME}-XXXXXX.${REGION}.run.app"
    echo -e "${YELLOW}[DRY RUN] Backend URL would be:${NC} ${BACKEND_URL}"
    echo ""
  fi
else
  echo -e "${BLUE}Step 3: Skip Backend Deployment${NC}"
  echo -e "${YELLOW}Note: Fetching existing backend URL...${NC}"

  if [[ "${DRY_RUN}" == "false" ]]; then
    BACKEND_URL=$(gcloud run services describe "${BACKEND_SERVICE_NAME}" \
      --platform=managed \
      --region="${REGION}" \
      --project="${PROJECT_ID}" \
      --format='value(status.url)' 2>/dev/null || echo "")

    if [[ -z "${BACKEND_URL}" ]]; then
      echo -e "${RED}✗ Backend service not found. Deploy backend first or use --backend-only${NC}"
      exit 1
    fi
    echo -e "${GREEN}✓${NC} Using existing backend: ${BACKEND_URL}"
  else
    BACKEND_URL="https://${BACKEND_SERVICE_NAME}-XXXXXX.${REGION}.run.app"
  fi
  echo ""
fi

# Step 4: Deploy App
if [[ "${DEPLOY_APP}" == "true" ]]; then
  echo -e "${BLUE}Step 4: Deploy Dashboard App${NC}"
  echo -e "${GREEN}▶${NC} Building and deploying dashboard container..."

  APP_DEPLOY_CMD=(
    gcloud run deploy "${APP_SERVICE_NAME}"
    --source="${APP_DIR}"
    --platform=managed
    --region="${REGION}"
    --project="${PROJECT_ID}"
    --set-env-vars="NEXT_PUBLIC_API_URL=${BACKEND_URL}"
    --memory=1Gi
    --cpu=1
    --timeout=60s
    --max-instances=10
    --min-instances=0
    --concurrency=80
    --port=3000
  )

  if [[ "${ALLOW_UNAUTHENTICATED}" == "true" ]]; then
    APP_DEPLOY_CMD+=(--allow-unauthenticated)
    echo -e "${YELLOW}Note: Dashboard allows unauthenticated access${NC}"
  else
    APP_DEPLOY_CMD+=(--no-allow-unauthenticated)
  fi

  run_cmd "${APP_DEPLOY_CMD[@]}"
  echo ""

  # Get app URL
  if [[ "${DRY_RUN}" == "false" ]]; then
    echo -e "${CYAN}Retrieving dashboard service URL...${NC}"
    APP_URL=$(gcloud run services describe "${APP_SERVICE_NAME}" \
      --platform=managed \
      --region="${REGION}" \
      --project="${PROJECT_ID}" \
      --format='value(status.url)')

    echo -e "${GREEN}✓${NC} Dashboard deployed successfully!"
    echo -e "${GREEN}Dashboard URL:${NC} ${APP_URL}"
    echo ""

    # Update backend CORS settings
    echo -e "${BLUE}Step 5: Update Backend CORS Settings${NC}"
    echo -e "${YELLOW}Note: You may need to update the backend CORS settings to allow the dashboard URL:${NC}"
    echo ""
    echo -e "${CYAN}Add this URL to the CORS allow_origins in backend/app/main.py:${NC}"
    echo -e "  ${APP_URL}"
    echo ""
  else
    APP_URL="https://${APP_SERVICE_NAME}-XXXXXX.${REGION}.run.app"
    echo -e "${YELLOW}[DRY RUN] Dashboard URL would be:${NC} ${APP_URL}"
    echo ""
  fi
else
  echo -e "${BLUE}Step 4: Skip App Deployment${NC}"
  echo ""
fi

# Final Summary
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}   Deployment Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [[ "${DRY_RUN}" == "false" ]]; then
  if [[ "${DEPLOY_BACKEND}" == "true" ]]; then
    echo -e "${CYAN}Backend Service:${NC}"
    echo -e "  URL: ${BACKEND_URL}"
    echo -e "  Endpoints:"
    echo -e "    - GET  ${BACKEND_URL}/"
    echo -e "    - GET  ${BACKEND_URL}/health"
    echo -e "    - GET  ${BACKEND_URL}/model/info"
    echo -e "    - POST ${BACKEND_URL}/predict"
    echo -e "    - WS   ${BACKEND_URL}/ws/predict"
    echo ""
  fi

  if [[ "${DEPLOY_APP}" == "true" ]]; then
    echo -e "${CYAN}Dashboard App:${NC}"
    echo -e "  URL: ${APP_URL}"
    echo ""
  fi

  echo -e "${CYAN}Next Steps:${NC}"
  echo "  1. Test the backend health endpoint:"
  echo "     curl ${BACKEND_URL}/health"
  echo ""
  echo "  2. Test the dashboard:"
  echo "     Open ${APP_URL} in your browser"
  echo ""
  echo "  3. Update backend CORS if needed (see backend/app/main.py)"
  echo ""

  if [[ "${ALLOW_UNAUTHENTICATED}" == "false" ]]; then
    echo -e "${YELLOW}Authentication Required:${NC}"
    echo "  Grant access to specific users or service accounts:"
    echo ""
    echo "  Backend:"
    echo -e "${BLUE}  gcloud run services add-iam-policy-binding ${BACKEND_SERVICE_NAME} \\${NC}"
    echo -e "${BLUE}    --region=${REGION} \\${NC}"
    echo -e "${BLUE}    --project=${PROJECT_ID} \\${NC}"
    echo -e "${BLUE}    --member='user:EMAIL' \\${NC}"
    echo -e "${BLUE}    --role='roles/run.invoker'${NC}"
    echo ""
    echo "  Dashboard:"
    echo -e "${BLUE}  gcloud run services add-iam-policy-binding ${APP_SERVICE_NAME} \\${NC}"
    echo -e "${BLUE}    --region=${REGION} \\${NC}"
    echo -e "${BLUE}    --project=${PROJECT_ID} \\${NC}"
    echo -e "${BLUE}    --member='user:EMAIL' \\${NC}"
    echo -e "${BLUE}    --role='roles/run.invoker'${NC}"
    echo ""
  fi

  # Save URLs to config file
  CONFIG_FILE="${PROJECT_ROOT}/.deployment-urls"
  echo "BACKEND_URL=${BACKEND_URL}" > "${CONFIG_FILE}"
  if [[ "${DEPLOY_APP}" == "true" ]]; then
    echo "APP_URL=${APP_URL}" >> "${CONFIG_FILE}"
  fi
  echo -e "${GREEN}✓${NC} Deployment URLs saved to ${CONFIG_FILE}"
  echo ""
fi

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
