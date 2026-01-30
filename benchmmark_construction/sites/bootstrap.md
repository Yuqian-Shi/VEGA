# 网站部署与初始化指南

本文档提供了框架支持的所有 Self-host 网站的部署和数据初始化步骤。每个网站都使用 Docker Compose 进行容器化部署，确保环境一致性和快速启动。

> **注意**: 请确保已安装 Docker 和 Docker Compose（v2+）

---

## VeOps CMDB

### 部署步骤

#### 1. 启动服务

进入 CMDB 目录并启动容器：

```bash
cd {$sites}/cmdb
docker compose up -d
```

#### 2. 验证部署

等待容器启动完成后，访问 Web 界面验证部署是否成功：

- **访问地址**: `http://localhost:9903`（根据实际配置调整）
- **验证方式**: 能够看到登录页面即表示部署成功

![CMDB 登录页面](assets/cmdb/landing.png)

### 数据初始化

#### 1. 登录系统

使用默认账户登录：

- **用户名**: `demo`
- **密码**: `123456`

![CMDB 登录界面](assets/cmdb/login.png)

#### 2. 验证初始化

登录成功后，进入首页查看初始化数据，确认系统已正确初始化：

![CMDB 首页](assets/cmdb/homepage.png)

---

## iTop

### 部署步骤

#### 1. 启动服务

进入 iTop 目录并启动容器：

```bash
cd {$sites}/iTop
docker compose up -d
```

#### 2. 验证部署

等待容器启动完成后，访问 Web 界面验证部署是否成功：

- **访问地址**: `http://localhost:9904`（根据实际配置调整）
- **验证方式**: 能够看到安装向导页面即表示部署成功

![iTop 安装向导](assets/itop/landing.png)

### 数据初始化

iTop 需要通过 Web 安装向导完成初始化配置，请按照以下步骤操作：

#### 1. 开始安装

点击 "Install a new iTop" 开始安装：

![iTop 安装选项](assets/itop/install_new_itop.png)

#### 2. 同意许可协议

阅读并同意 iTop 许可协议：

![iTop 许可协议](assets/itop/license_agreement.png)

#### 3. 配置数据库连接

填写数据库连接信息：

- **Server Name**: `db`
- **Login**: `itop`
- **Password**: `It0pDbP@ss!`

![iTop 数据库配置](assets/itop/database_configuration.png)

#### 4. 设置管理员账户

设置管理员账户密码：

- **密码**: `admin1234`

![iTop 管理员账户设置](assets/itop/administrator_account.png)

#### 5. 其他参数配置

保持默认配置或根据需要进行调整：

![iTop 其他参数配置](assets/itop/miscellaneous_parameters.png)

#### 6. 配置管理选项

选择配置管理相关功能模块：

![iTop 配置管理选项](assets/itop/configuration_management_options.png)

#### 7. 服务管理选项

选择服务管理相关功能模块：

![iTop 服务管理选项](assets/itop/service_management_options.png)

#### 8. 工单管理选项

选择工单管理相关功能模块：

![iTop 工单管理选项](assets/itop/tickets_management_options.png)

#### 9. 变更管理选项

选择变更管理相关功能模块：

![iTop 变更管理选项](assets/itop/change_management_options.png)

#### 10. ITIL 工单选项

选择额外的 ITIL 工单功能：

![iTop ITIL 工单选项](assets/itop/additional_ITIL_tickets.png)

#### 11. 准备安装

确认所有配置信息，准备开始安装：

![iTop 准备安装](assets/itop/ready_to_install.png)

![iTop 准备安装详情](assets/itop/ready_to_install2.png)

#### 12. 完成安装

等待安装完成后，系统会显示安装成功页面：

![iTop 安装完成](assets/itop/done.png)

#### 13. 验证初始化

使用管理员账户登录系统，进入首页查看初始化数据，确认系统已正确初始化：

- **用户名**: `admin`
- **密码**: `admin1234`

![iTop 首页](assets/itop/homepage.png)

---

## OpenProject

### 部署步骤

#### 1. 启动服务

进入 OpenProject 目录并启动容器：

```bash
cd {$sites}/openproject
docker compose up -d
```

#### 2. 验证部署

等待容器启动完成后，访问 Web 界面验证部署是否成功：

- **访问地址**: `http://localhost:9902`（根据实际配置调整）
- **验证方式**: 能够看到登录页面即表示部署成功

![OpenProject 登录页面](assets/openproject/landing.png)

### 数据初始化

#### 1. 首次登录

使用默认管理员账户登录：

- **用户名**: `admin`
- **密码**: `admin`

![OpenProject 登录界面](assets/openproject/login.png)

#### 2. 重置密码

首次登录后，系统会要求重置密码。设置新密码：

- **新密码**: `admin123456`

![OpenProject 重置密码](assets/openproject/reset_password.png)

#### 3. 验证初始化

使用新密码登录后，进入首页查看初始化数据，确认系统已正确初始化：

![OpenProject 首页](assets/openproject/homepage.png)

---

## ZenTao（禅道）

### 部署步骤

#### 1. 启动服务

进入 ZenTao 目录并启动容器：

```bash
cd {$sites}/zentao
docker compose up -d
```

#### 2. 验证部署

等待容器启动完成后，访问 Web 界面验证部署是否成功：

- **访问地址**: `http://localhost:9901`（根据实际配置调整）
- **验证方式**: 能够看到安装向导页面即表示部署成功

![ZenTao 安装向导](assets/zentao/landing.png)

### 数据初始化

ZenTao 需要通过 Web 安装向导完成初始化配置，请按照以下步骤操作：

#### 1. 同意许可协议

阅读并同意 ZenTao 许可协议：

![ZenTao 许可协议](assets/zentao/license_agreement.png)

#### 2. 系统检测

系统会自动检测环境配置，默认点击"下一步"继续：

![ZenTao 系统检测](assets/zentao/system_checker.png)

#### 3. 配置创建检测

系统会检测配置文件创建情况，默认点击"下一步"继续：

![ZenTao 配置创建检测](assets/zentao/prop_created.png)

#### 4. 完成基础配置

基础配置完成后，点击"完成"进入下一步：

![ZenTao 配置完成](assets/zentao/done.png)

#### 5. 选择管理模式

选择"使用全生命周期管理模式"：

![ZenTao 管理模式选择](assets/zentao/use_type.png)

#### 6. 设置账户并导入 Demo 数据

在账户设置页面，**务必勾选"导入 demo 数据"**，以便后续评测使用：

![ZenTao 账户设置](assets/zentao/set_account.png)

#### 7. 验证初始化

安装完成后，进入首页查看初始化数据，确认系统已正确初始化：

![ZenTao 首页](assets/zentao/homepage.png)
