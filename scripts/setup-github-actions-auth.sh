#!/bin/bash
#
# Setup Workload Identity Federation for GitHub Actions
# This allows GitHub Actions to authenticate to GCP without service account keys
#

set -euo pipefail

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-coderd}"
REGION="us-central1"
SERVICE_ACCOUNT_NAME="github-actions-deployer"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
WORKLOAD_IDENTITY_POOL="github-actions-pool"
WORKLOAD_IDENTITY_PROVIDER="github-provider"
GITHUB_REPO="${GITHUB_REPO:-VectorInstitute/gaca-early-warning}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   GitHub Actions Workload Identity Setup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Project ID:       $PROJECT_ID"
echo "Region:           $REGION"
echo "Service Account:  $SERVICE_ACCOUNT_EMAIL"
echo "GitHub Repo:      $GITHUB_REPO"
echo ""

# Set project
echo -e "${BLUE}Step 1: Setting GCP project${NC}"
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo -e "${BLUE}Step 2: Enabling required APIs${NC}"
gcloud services enable \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  cloudresourcemanager.googleapis.com \
  sts.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com

# Create service account
echo -e "${BLUE}Step 3: Creating service account${NC}"
if gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" &>/dev/null; then
  echo -e "${YELLOW}Service account already exists${NC}"
else
  gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
    --display-name="GitHub Actions Deployer" \
    --description="Service account for GitHub Actions to deploy to Cloud Run"
  echo -e "${GREEN}✓${NC} Service account created"
fi

# Grant permissions
echo -e "${BLUE}Step 4: Granting IAM permissions${NC}"
ROLES=(
  "roles/run.admin"                    # Deploy Cloud Run services
  "roles/iam.serviceAccountUser"       # Act as service account
  "roles/artifactregistry.writer"      # Push to Artifact Registry
  "roles/storage.objectViewer"         # Read from Cloud Storage (for logs)
  "roles/logging.viewer"               # View logs
)

for role in "${ROLES[@]}"; do
  echo "  Granting $role..."
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="$role" \
    --condition=None \
    --quiet
done
echo -e "${GREEN}✓${NC} Permissions granted"

# Create Workload Identity Pool
echo -e "${BLUE}Step 5: Creating Workload Identity Pool${NC}"
if gcloud iam workload-identity-pools describe "$WORKLOAD_IDENTITY_POOL" \
  --location=global &>/dev/null; then
  echo -e "${YELLOW}Workload Identity Pool already exists${NC}"
else
  gcloud iam workload-identity-pools create "$WORKLOAD_IDENTITY_POOL" \
    --location=global \
    --display-name="GitHub Actions Pool" \
    --description="Workload Identity Pool for GitHub Actions"
  echo -e "${GREEN}✓${NC} Workload Identity Pool created"
fi

# Create Workload Identity Provider
echo -e "${BLUE}Step 6: Creating Workload Identity Provider${NC}"
if gcloud iam workload-identity-pools providers describe "$WORKLOAD_IDENTITY_PROVIDER" \
  --workload-identity-pool="$WORKLOAD_IDENTITY_POOL" \
  --location=global &>/dev/null; then
  echo -e "${YELLOW}Workload Identity Provider already exists${NC}"
else
  gcloud iam workload-identity-pools providers create-oidc "$WORKLOAD_IDENTITY_PROVIDER" \
    --workload-identity-pool="$WORKLOAD_IDENTITY_POOL" \
    --location=global \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
    --attribute-condition="assertion.repository_owner == '${GITHUB_REPO%%/*}'"
  echo -e "${GREEN}✓${NC} Workload Identity Provider created"
fi

# Allow GitHub Actions to impersonate service account
echo -e "${BLUE}Step 7: Binding service account to Workload Identity${NC}"
gcloud iam service-accounts add-iam-policy-binding "$SERVICE_ACCOUNT_EMAIL" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')/locations/global/workloadIdentityPools/$WORKLOAD_IDENTITY_POOL/attribute.repository/$GITHUB_REPO"
echo -e "${GREEN}✓${NC} Service account bound to GitHub repository"

# Get Workload Identity Provider resource name
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
WIF_PROVIDER="projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$WORKLOAD_IDENTITY_POOL/providers/$WORKLOAD_IDENTITY_PROVIDER"

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}   Setup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}GitHub Secrets Configuration${NC}"
echo ""
echo "Add these secrets to your GitHub repository:"
echo "  Settings → Secrets and variables → Actions → New repository secret"
echo ""
echo -e "${YELLOW}Required Secrets:${NC}"
echo ""
echo "Secret Name: GCP_PROJECT_ID"
echo "Value: $PROJECT_ID"
echo ""
echo "Secret Name: GCP_SERVICE_ACCOUNT"
echo "Value: $SERVICE_ACCOUNT_EMAIL"
echo ""
echo "Secret Name: WIF_PROVIDER"
echo "Value: $WIF_PROVIDER"
echo ""
echo "Secret Name: MAPBOX_TOKEN"
echo "Value: <your-mapbox-token-from-app/.env.local>"
echo ""
echo -e "${BLUE}Verification${NC}"
echo ""
echo "To verify the setup, you can manually test authentication:"
echo ""
echo "  gcloud iam workload-identity-pools create-cred-config \\"
echo "    $WIF_PROVIDER \\"
echo "    --service-account=$SERVICE_ACCOUNT_EMAIL \\"
echo "    --output-file=credentials.json"
echo ""
