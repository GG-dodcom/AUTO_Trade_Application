## Title: Optimizing Returns and Risk Management in Automated Trading Using Reinforcement Learning with Proximal Policy Optimization

### Abstract
This study presents the development of an automated trading system for the XAUUSD (gold) CFD market, leveraging Proximal Policy Optimization (PPO) reinforcement learning to address the challenges faced by retail investors, including high leverage, market volatility, and limited access to advanced trading tools. Through comprehensive literature reviews and a survey of 32 retail traders, the system integrates historical price data, technical indicators, and economic calendar events to enable data-driven trading decisions. The methodology employs dynamic data processing and risk-aware reward mechanisms to optimize profitability while mitigating risks. The system achieved a 12.2% cumulative return, though a 38.98% drawdown highlights the need for enhanced risk management. Aligned with SDG 10 (Reduced Inequalities), the project democratizes access to sophisticated trading technology via a user-friendly desktop application, fostering financial inclusion for retail investors. Future improvements include multi-market support and advanced risk controls to enhance stability and accessibility.

Keywords: Automated Trading, Reinforcement Learning, CFD Trading, Financial Inclusion, SDG 10.

### List of technologies
#### Programming Languages & Frameworks
1. Python
	- Used for:
		- Model training (Reinforcement Learning with PPO)
		- Backend development
		- Integration with MetaTrader 5 (via Python API)
	- Benefits: Readability, large ecosystem, scientific and ML support
2. TypeScript
	- Used with Next.js for type-safe frontend/backend development
3. Next.js (React-based framework)
	- Used for:
		- Frontend and backend in the same framework
		- Server-Side Rendering (SSR) & Client-Side Rendering (CSR)
		- Real-time UI updates for trading data
		- Integrated with Electron to create a desktop app
4. Electron
	- Used to package the Next.js application as a cross-platform desktop app
	- Target platforms: Windows, macOS, Linux

#### Machine Learning / Deep Learning
1. Reinforcement Learning Algorithm: PPO (Proximal Policy Optimization)
	- Used to train the automated trading agent
2. Deep Learning Frameworks
	- TensorFlow
	- PyTorch
	- Both used to build and train the RL models
3. FinRL
	- A Python-based RL framework tailored for finance
	- Used for building/testing trading strategies in financial environments

#### Trading Platform Integration
MetaTrader 5 (MT5)
- Used for real-time market data, executing trades, and account management
- Integrated via official Python API
- Fully supported on Windows 10+ only

#### Database Technology
SQLite
- Embedded, lightweight database for local storage
- Features:
	- Fast read/write
	- Cross-platform support
	- Supports structured & semi-structured data (JSON)
	- Secure (with SQLCipher)
	- Write-Ahead Logging (WAL) for concurrency handling

#### Development Tools
- Visual Studio Code (VS Code) – Primary code editor for writing TypeScript, and Next.js code
- Jupyter Notebook
- Used for:
	- Interactive model development
	- Step-by-step debugging
	- Visualizing RL training results

#### Operating System
- Windows 10 or later (64-bit)
	- Required for: Full compatibility with MetaTrader 5 Python API

#### Hardware Requirements for Model Training
- 16 GB RAM
- Multi-core CPU
- 4060TI NVIDIA GPU (for RL training with TensorFlow/PyTorch)
- 500 GB SSD

### Interface Design
Sign Up
<img width="975" height="542" alt="image" src="https://github.com/user-attachments/assets/37435147-89a7-4122-9292-862c67476b17" />
Figure 1: Sign up interface design.

Login
<img width="975" height="516" alt="image" src="https://github.com/user-attachments/assets/87a8e0b2-528b-486d-9658-8ddc757b7e92" />
Figure 2: Login interface design. 

Login Broker Account
<img width="975" height="680" alt="image" src="https://github.com/user-attachments/assets/5d8bc9bc-ecfd-4399-8c46-0977dc3c6d78" />
Figure 3: Login broker account interface design.
 
Dashboard
<img width="958" height="644" alt="image" src="https://github.com/user-attachments/assets/40ddc4a9-eea9-41e3-9675-6628c1c9f5e1" />
<img width="961" height="655" alt="image" src="https://github.com/user-attachments/assets/038d1122-9c19-4488-adc8-33bad99aa5b6" />
<img width="804" height="627" alt="image" src="https://github.com/user-attachments/assets/60564d14-92ba-4ddc-af76-f1a9a647b9c0" />
<img width="812" height="667" alt="image" src="https://github.com/user-attachments/assets/ddb5633c-967a-4c1b-bb59-ed58da60703f" />
Figure 4: Dashboard interface design.

Trading Journal
<img width="975" height="807" alt="image" src="https://github.com/user-attachments/assets/d6b75dbe-fa20-4658-818e-b23b433d176d" />
Figure 5: Trading journal interface design.

Monthly Report
<img width="975" height="658" alt="image" src="https://github.com/user-attachments/assets/865662e9-e080-4d01-889f-50081510cde5" />
<img width="975" height="340" alt="image" src="https://github.com/user-attachments/assets/7805044f-90e4-4dc2-815f-4dec28d38c18" />
Figure 6: Monthly report interface design. 

Backtest Report 
<img width="975" height="830" alt="image" src="https://github.com/user-attachments/assets/c316b371-b521-41b1-98c3-2e3c5d8d120e" />
Figure 7: Backtest report interface design.

Setting
<img width="975" height="623" alt="image" src="https://github.com/user-attachments/assets/bceb506f-6e2f-4328-a1ed-d22f3e029dc7" />
<img width="975" height="615" alt="image" src="https://github.com/user-attachments/assets/0441bd0f-dbe5-45c4-9901-f2df27a5abfd" />
<img width="975" height="622" alt="image" src="https://github.com/user-attachments/assets/78429940-7c8a-4991-89d8-f8d23012acf4" />
Figure 8: Setting interface design.

### User Manual
This user manual provides step-by-step instructions to set up and operate the AUTO Trade application for automated trading on the XAUUSD market using MetaTrader 5 (MT5). Follow these steps carefully to ensure a smooth experience. Each step includes references to figures for visual guidance and troubleshooting tips where applicable.

#### Prerequisites
- A stable internet connection.
- A Windows PC (for running Auto Trade.exe).
- Basic familiarity with trading platforms and account management.

#### Step-by-Step Instructions
##### Step 1: Install MetaTrader5
1. Visit [https://www.metatrader5.com/en/download](https://www.metatrader5.com/en/download).
2. Download the MetaTrader 5 platform for Windows.
3. Run the installer and follow the on-screen instructions to complete the installation.

<img width="1013" height="422" alt="image" src="https://github.com/user-attachments/assets/c0466553-9f1e-4463-a3b0-39aeaf0a12d4" />

Figure 224: Download MetaTrader5 platform.

_**Troubleshooting Tip**_: If the download fails, check your internet connection or try a different browser. Ensure your system meets MT5’s minimum requirements (Windows 7 or later).

##### Step 2: Register a Demo Account
1.	Navigate to https://en.octafxmy.net/.
2.	Select the option to create a demo account for MetaTrader5.
3.	Fill in the required details (e.g., name, email, and preferred currency).
4.	Save the account credentials (account number, password, and server) provided upon registration.

<img width="487" height="705" alt="image" src="https://github.com/user-attachments/assets/2a8b42f8-273d-48c4-88ee-a5790de1f8fe" />

Figure 225: Create a demo account for MetaTrader5 platform.

<img width="621" height="678" alt="image" src="https://github.com/user-attachments/assets/14f7c748-b9a9-4538-a328-7ed03383dcbf" />

Figure 226: Save this account number, server and password to login your MT5 broker account.

_**Note**_: A demo account allows you to test the trading system without financial risk. Keep your credentials secure for future use.

##### Step 3: Login to MetaTrader5
1.	Launch the MetaTrader5 application. 
2.	Enter your demo account credentials (account number, password, and server). 
3.	Verify the login by checking the connection status in the bottom-right corner of MT5 (should display “Connected”).

<img width="555" height="186" alt="image" src="https://github.com/user-attachments/assets/d4d906a5-9d5d-4ed8-b6e3-e6130f5b9318" />

Figure 227: Login to MetaTrader5.

_**Troubleshooting Tip**_: If login fails, ensure the server name matches exactly as provided by OctaFX. Contact your broker’s support if issues persist.

##### Step 4: Enable Algorithmic Trading
1.	In MetaTrader 5, locate the “Algo Trading” button in the toolbar. 
2.	Click the button to enable algorithmic trading (the button will turn green when activated).
 
<img width="637" height="141" alt="image" src="https://github.com/user-attachments/assets/a33aa158-abe6-469a-aab0-94537b27930b" />

Figure 228: Enable algorithm trade on MetaTrader5.

_**Important**_: Algorithmic trading must be enabled for the AUTO Trade application to execute trades. If the run model controller button remains gray, check MT5’s toolbar to enable the ‘Algo Trading’.

<img width="344" height="513" alt="image" src="https://github.com/user-attachments/assets/692ac72b-f8f7-4a5a-a6db-d12e307f6ae2" />

Figure 229: Run model controller will show in gray if ‘Algo Trading’ is disable.

##### Step 5: Install and Launch AUTO Trade Application
1.	Download the AUTO Trade application zip file from the provided source. 
2.	Extract the zip file to a preferred location on your computer. 
3.	Navigate to the extracted folder and double-click Auto Trade.exe to launch the application.

<img width="319" height="47" alt="image" src="https://github.com/user-attachments/assets/662ae484-2ddf-49a2-933d-b54594260613" />

Figure 230: Unzip folder.

<img width="975" height="166" alt="image" src="https://github.com/user-attachments/assets/a179d832-f37c-4fb6-b105-d525b0992a36" />

Figure 231: Double click on ‘Auto Trade’ application to open it.

##### Step 6: Sign Up for AUTO Trade
1.	On the application’s startup screen, click “Sign Up.” 
2.	Enter a valid email address and a password. 
3.	Submit the form to create your account.

<img width="975" height="740" alt="image" src="https://github.com/user-attachments/assets/72999a53-a64c-42a8-b623-b0f54ea9e7f9" />

Figure 232: Frontend signup page.

_**Note**_: If you encounter an error (e.g., “Email already in use”), try a different email.

##### Step 7: Login to AUTO Trade
1.	Return to the login page. 
2.	Enter your registered email and password. 
3.	Click “Login” to access the application.

<img width="975" height="738" alt="image" src="https://github.com/user-attachments/assets/06e395d4-2d04-457d-a9dc-6a7a08183a2f" />

Figure 233: Frontend login page.

##### Step 8: Connect AUTO Trade to MetaTrader5
1.	In the AUTO Trade application, navigate to the MT5 account setup section. 
2.	Enter your MT5 demo account details (account number, password, and server). 
3.	Optionally, check “Save Password” to securely store your credentials in the application’s database for automatic retrieval in future sessions. 
4.	Submit the form. If successful, MetaTrader5 will automatically open, confirming the connection.

<img width="831" height="628" alt="image" src="https://github.com/user-attachments/assets/242d87c4-acfd-4e01-b41e-38817c5ec952" />

Figure 234: Fill up the detail to initial MetaTrader5 platform.

<img width="830" height="622" alt="image" src="https://github.com/user-attachments/assets/7c94ab31-0d2b-477e-9fe4-fcad93397615" />

Figure 235: If ‘Save password’ checked, it will auto retrieve MT5 account detail next time.

<img width="925" height="555" alt="image" src="https://github.com/user-attachments/assets/2e464fc1-53f3-4d0b-8d53-053e902eebe4" />

Figure 236: MT5 auto-opens after successful login to MT5 broker account.

##### Step 9: Run the AI Trading Model
1.	In the AUTO Trade dashboard, locate the “Play” button for the AI auto-trade model. 
2.	Click the “Play” button to start the model. The status will update to “Running.” 
3.	To stop the model, click the “Stop” button.

<img width="439" height="297" alt="image" src="https://github.com/user-attachments/assets/29b8ce89-fa7f-40e5-9c35-5e9d8836e529" />

Figure 237: Run model controller stop & start.

_**Note**_: The AI model uses a PPO-based reinforcement learning algorithm to execute trades. Monitor its performance to ensure it aligns with your trading goals.

##### Step 10: Monitor and Analyze Trading Performance
1. Navigate to the “Backtest Report” to review the AI model’s historical performance metrics (e.g., cumulative return, drawdown). 
2. Use the “Dashboard” to view real-time trade updates, including charts and notifications. 
3. Access the “Trading Journal” to track individual trade details, such as entry/exit times and profit/loss. 
4. Visit the “Monthly Report” for aggregated performance statistics by month. 
5. In the “Settings” page, customize trading parameters: 
	1. **Symbol**: Specific XAUUSD symbol name assigned by the broker account.
	2. **Volume**: Set the trade size (e.g., 0.1 lots).
	3. **Stop Loss**: Define the maximum loss per trade in points.
	4. **Take Profit**: Set the target profit per trade in points.

<img width="975" height="640" alt="image" src="https://github.com/user-attachments/assets/d3196a69-ed76-46f8-8863-dcde05c4a2cf" />

Figure 238: Frontend backtest report page.

<img width="869" height="604" alt="image" src="https://github.com/user-attachments/assets/1de8e35e-8809-4633-87c6-0d513e08eafc" />

Figure 239: Frontend dashboard pages.

<img width="831" height="602" alt="image" src="https://github.com/user-attachments/assets/279fb0af-1e6a-41a4-9c35-5a247fb16ce9" />

Figure 240: Frontend trading journal page.

<img width="905" height="672" alt="image" src="https://github.com/user-attachments/assets/22a3d559-86aa-4b8d-8d58-e855c3ff4ccb" />

Figure 241: Frontend monthly report page.

<img width="873" height="566" alt="image" src="https://github.com/user-attachments/assets/9354bc46-a764-4ada-9620-db5a17d87e62" />

Figure 242: Frontend setting page.

