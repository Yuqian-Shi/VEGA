# Website Deployment and Initialization Guide

This document provides the deployment and data initialization steps for all self-hosted websites supported by the framework. Each website uses Docker Compose for containerized deployment, ensuring environmental consistency and quick startup.

> **Note**: Please ensure Docker and Docker Compose (v2+) are installed.

---

## VeOps CMDB

### Deployment Steps

#### 1. Start Services

Navigate to the CMDB directory and start the containers:

```bash
cd {$sites}/cmdb
docker compose up -d
```

#### 2. Verify Deployment

After the containers have started, access the web interface to verify successful deployment:

- **Access Address**: `http://localhost:9903`（adjust based on actual configuration）
- **Verification Method**: Seeing the login page indicates successful deployment.

![CMDB Login](assets/cmdb/landing.png)

### Data Initialization

#### 1. Log in to the System

Use the default account to log in:

- **Username**: `demo`
- **Password**: `123456`

![CMDB Login](assets/cmdb/login.png)

#### 2. Verify Initialization

After successful login, access the homepage to view the initialized data and confirm the system is correctly initialized:

![CMDB Homepage](assets/cmdb/homepage.png)

---

## iTop

### Deployment Steps

#### 1. Start Services

Navigate to the iTop directory and start the containers:

```bash
cd {$sites}/iTop
docker compose up -d
```

#### 2. Verify Deployment

After the containers have started, access the web interface to verify successful deployment:

- **Access Address**: `http://localhost:9904`（adjust based on actual configuration）
- **Verification Method**: Seeing the installation wizard page indicates successful deployment.

![iTop installation wizard page](assets/itop/landing.png)

### Data Initialization

iTop needs to be initialized through the web installation wizard. Please follow the steps below:

#### 1. Start Installation

Click "Install a new iTop" to begin the installation:

![iTop install options](assets/itop/install_new_itop.png)

#### 2. Accept License Agreement

Read and accept the iTop license agreement:

![iTop license agreement](assets/itop/license_agreement.png)

#### 3. Configure Database Connection

Fill in the database connection information:

- **Server Name**: `db`
- **Login**: `itop`
- **Password**: `It0pDbP@ss!`

![iTop database configuration](assets/itop/database_configuration.png)

#### 4. Set Administrator Account

Set the administrator account password:

- **Password**: `admin1234`

![iTop administrator account](assets/itop/administrator_account.png)

#### 5. Other Configurations

Keep the default configuration or adjust as needed:

![iTop other configurations](assets/itop/miscellaneous_parameters.png)

#### 6. Configuration Management Options

Select relevant functional modules for configuration management:

![iTop configuration management options](assets/itop/configuration_management_options.png)

#### 7. Service Management Options

Select relevant functional modules for service management:

![iTop service management options](assets/itop/service_management_options.png)

#### 8. Tickets Management Options

Select relevant functional modules for ticket management:

![iTop ticket management options](assets/itop/tickets_management_options.png)

#### 9. Change Management Options

Select relevant functional modules for change management:

![iTop change management options](assets/itop/change_management_options.png)

#### 10. ITIL Tickets Options

Select additional ITIL ticket functions:

![iTop ITIL ticket options](assets/itop/additional_ITIL_tickets.png)

#### 11. Ready to Install

Confirm all configuration information and prepare to start the installation:

![iTop ready install](assets/itop/ready_to_install.png)

![iTop install details](assets/itop/ready_to_install2.png)

#### 12. Complete Installation

After the installation is complete, the system will display a success page:

![iTop complete installation](assets/itop/done.png)

#### 13. Verify Initialization

Log in to the system using the administrator account and access the homepage to view the initialized data, confirming the system is correctly initialized:

- **Username**: `admin`
- **Password**: `admin1234`

![iTop Homepage](assets/itop/homepage.png)

---

## OpenProject

### Deployment Steps

#### 1. Start Services

Navigate to the OpenProject directory and start the containers:

```bash
cd {$sites}/openproject
docker compose up -d
```

#### 2. Verify Deployment

After the containers have started, access the web interface to verify successful deployment:

- **Access Address**: `http://localhost:9902`（adjust based on actual configuration）
- **Verification Method**: Seeing the login page indicates successful deployment.

![OpenProject login page](assets/openproject/landing.png)

### Data Initialization

#### 1. First Login

Use the default administrator account to log in:

- **Username**: `admin`
- **Pssword**: `admin`

![OpenProject login page](assets/openproject/login.png)

#### 2. Reset Password

After the first login, the system will require a password reset. Set a new password:

- **New Password**: `admin123456`

![OpenProject reset password](assets/openproject/reset_password.png)

#### 3. Verify Initialization

After logging in with the new password, access the homepage to view the initialized data and confirm the system is correctly initialized:

![OpenProject Homepage](assets/openproject/homepage.png)

---

## ZenTao

### Deployment Steps

#### 1. Start Services

Navigate to the ZenTao directory and start the containers:

```bash
cd {$sites}/zentao
docker compose up -d
```

#### 2. Verify Deployment

After the containers have started, access the web interface to verify successful deployment:

- **Access Address**: `http://localhost:9901`（adjust based on actual configuration）
- **Verification Method**: Seeing the installation wizard page indicates successful deployment.

![ZenTao installation wizard page](assets/zentao/landing.png)

### Data Initialization

ZenTao needs to be initialized through the web installation wizard. Please follow the steps below:

#### 1. Accept License Agreement

Read and accept the ZenTao license agreement:

![ZenTao license agreement](assets/zentao/license_agreement.png)

#### 2. System Check

he system will automatically check the environment configuration. Click "Next" by default to continue:

![ZenTao system check](assets/zentao/system_checker.png)

#### 3. Configuration File Check

The system will check the creation status of the configuration file. Click "Next" by default to continue:

![ZenTao configuration file check](assets/zentao/prop_created.png)

#### 4. Complete Basic Configuration

After completing the basic configuration, click "Done" to proceed to the next step:

![ZenTao configuration done](assets/zentao/done.png)

#### 5. Select Management Mode

Select "Use full lifecycle management mode":

![ZenTao management mode](assets/zentao/use_type.png)

#### 6. Set Account and Import Demo Data

On the account settings page, be sure to check "Import demo data" for subsequent evaluation use:

![ZenTao set account](assets/zentao/set_account.png)

#### 7. Verify Initialization

After installation, access the homepage to view the initialized data and confirm the system is correctly initialized:

![ZenTao Homepage](assets/zentao/homepage.png)
