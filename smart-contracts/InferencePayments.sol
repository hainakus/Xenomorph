// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title InferencePayments
 * @dev Handles USDT payments for AI model inference queries
 * @dev Payment distribution: 70% seed nodes, 20% training miners, 10% treasury
 */
contract InferencePayments is ReentrancyGuard, Ownable {
    using SafeERC20 for IERC20;

    IERC20 public immutable usdtToken;
    
    uint256 public constant SEED_SHARE = 70;
    uint256 public constant TRAINING_SHARE = 20;
    uint256 public constant TREASURY_SHARE = 10;
    uint256 public constant TOTAL_SHARE = 100;
    
    uint256 public constant QUERY_PAYMENT_AMOUNT = 100 * 10**6; // 100 USDT
    uint256 public constant MIN_PAYMENT = 1 * 10**6; // 1 USDT
    
    mapping(bytes32 => QueryPayment) public queryPayments;
    mapping(address => uint256) public seedNodeEarnings;
    mapping(address => uint256) public trainingRewards;
    uint256 public treasuryBalance;
    
    address public treasuryWallet;
    address public seedNodePool;
    address public trainingPool;
    
    uint256 public totalQueriesProcessed;
    uint256 public totalPayoutVolume;
    
    // Payment tiers based on model complexity
    mapping(string => uint256) public modelPaymentRates;
    
    // Service fees
    uint256 public serviceFeeRate = 30; // 3% service fee
    uint256 public constant MAX_SERVICE_FEE = 500; // 50% max
    
    // Emergency pause
    bool public paused;
    
    event QueryPaymentInitiated(bytes32 indexed queryId, address indexed payer, uint256 amount);
    event QueryPaymentCompleted(bytes32 indexed queryId, uint256 seedAmount, uint256 trainingAmount, uint256 treasuryAmount);
    event QueryPaymentRefunded(bytes32 indexed queryId, address indexed payer, uint256 amount);
    event EarningsWithdrawn(address indexed recipient, uint256 amount);
    event TreasuryUpdated(address indexed newTreasury);
    event SeedNodePoolUpdated(address indexed newPool);
    event TrainingPoolUpdated(address indexed newPool);
    event PaymentRateUpdated(string indexed modelId, uint256 newRate);
    event Paused(bool indexed newPaused);
    
    struct QueryPayment {
        address payer;
        uint256 amount;
        uint256 timestamp;
        bool completed;
        bool refunded;
        bytes32 modelId;
    }
    
    error PaymentTooLow();
    error PaymentAlreadyCompleted();
    error PaymentNotFound();
    error InvalidRecipient();
    error ContractPaused();
    error InvalidServiceFee();
    error InsufficientBalance();
    error TransferFailed();
    error Unauthorized();
    
    modifier onlyWhenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }
    
    modifier onlyAdminOrTreasury() {
        require(msg.sender == owner() || msg.sender == treasuryWallet, "Unauthorized");
        _;
    }
    
    constructor(
        address _usdtToken,
        address _treasuryWallet,
        address _seedNodePool,
        address _trainingPool
    ) Ownable(msg.sender) {
        require(_usdtToken != address(0), "Invalid USDT token");
        require(_treasuryWallet != address(0), "Invalid treasury");
        require(_seedNodePool != address(0), "Invalid seed pool");
        require(_trainingPool != address(0), "Invalid training pool");
        
        usdtToken = IERC20(_usdtToken);
        treasuryWallet = _treasuryWallet;
        seedNodePool = _seedNodePool;
        trainingPool = _trainingPool;
        
        emit TreasuryUpdated(_treasuryWallet);
        emit SeedNodePoolUpdated(_seedNodePool);
        emit TrainingPoolUpdated(_trainingPool);
    }
    
    /**
     * @dev Initiate a payment for a query
     * @param queryId Unique identifier for the query
     * @param modelId Identifier of the model being queried
     * @param amount Payment amount in USDT (6 decimals)
     */
    function initiatePayment(
        bytes32 queryId,
        string calldata modelId,
        uint256 amount
    ) external onlyWhenNotPaused nonReentrant {
        require(amount >= MIN_PAYMENT, PaymentTooLow());
        
        // Check custom payment rate for specific model
        uint256 requiredAmount = modelPaymentRates[modelId];
        if (requiredAmount > 0) {
            require(amount >= requiredAmount, PaymentTooLow());
        } else {
            require(amount >= QUERY_PAYMENT_AMOUNT, PaymentTooLow());
        }
        
        // Transfer USDT from payer to contract
        usdtToken.safeTransferFrom(msg.sender, address(this), amount);
        
        // Store payment details
        queryPayments[queryId] = QueryPayment({
            payer: msg.sender,
            amount: amount,
            timestamp: block.timestamp,
            completed: false,
            refunded: false,
            modelId: bytes32(bytes(modelId))
        });
        
        totalQueriesProcessed++;
        totalPayoutVolume += amount;
        
        emit QueryPaymentInitiated(queryId, msg.sender, amount);
    }
    
    /**
     * @dev Complete payment distribution after successful inference
     * @param queryId Query identifier
     */
    function completePayment(bytes32 queryId) external onlyAdminOrTreasury onlyWhenNotPaused nonReentrant {
        QueryPayment storage payment = queryPayments[queryId];
        require(payment.amount > 0, PaymentNotFound());
        require(!payment.completed, PaymentAlreadyCompleted());
        require(!payment.refunded, PaymentAlreadyCompleted());
        
        uint256 amount = payment.amount;
        
        // Calculate service fee
        uint256 serviceFee = (amount * serviceFeeRate) / 10000;
        uint256 distributableAmount = amount - serviceFee;
        
        // Distribute shares
        uint256 seedAmount = (distributableAmount * SEED_SHARE) / TOTAL_SHARE;
        uint256 trainingAmount = (distributableAmount * TRAINING_SHARE) / TOTAL_SHARE;
        uint256 treasuryAmount = distributableAmount - seedAmount - trainingAmount;
        
        // Transfer shares
        bool seedSuccess = _transferSafe(seedNodePool, seedAmount);
        bool trainingSuccess = _transferSafe(trainingPool, trainingAmount);
        bool treasurySuccess = _transferSafe(treasuryWallet, treasuryAmount);
        
        require(seedSuccess && trainingSuccess && treasurySuccess, TransferFailed());
        
        // Update balances
        seedNodeEarnings[seedNodePool] += seedAmount;
        trainingRewards[trainingPool] += trainingAmount;
        treasuryBalance += treasuryAmount;
        
        // Mark as completed
        payment.completed = true;
        
        emit QueryPaymentCompleted(queryId, seedAmount, trainingAmount, treasuryAmount);
    }
    
    /**
     * @dev Refund payment if inference fails
     * @param queryId Query identifier
     */
    function refundPayment(bytes32 queryId) external onlyAdminOrTreasury onlyWhenNotPaused nonReentrant {
        QueryPayment storage payment = queryPayments[queryId];
        require(payment.amount > 0, PaymentNotFound());
        require(!payment.completed && !payment.refunded, PaymentAlreadyCompleted());
        
        uint256 amount = payment.amount;
        address payer = payment.payer;
        
        // Mark as refunded before transfer to prevent reentrancy
        payment.refunded = true;
        
        // Refund full amount
        bool success = _transferSafe(payer, amount);
        require(success, TransferFailed());
        
        emit QueryPaymentRefunded(queryId, payer, amount);
    }
    
    /**
     * @dev Withdraw earnings for seed node pool
     * @param recipient Address to receive earnings
     * @param amount Amount to withdraw
     */
    function withdrawSeedEarnings(address recipient, uint256 amount) external onlyWhenNotPaused {
        require(recipient != address(0), InvalidRecipient());
        require(amount <= seedNodeEarnings[seedNodePool], InsufficientBalance());
        
        seedNodeEarnings[seedNodePool] -= amount;
        bool success = _transferSafe(recipient, amount);
        require(success, TransferFailed());
        
        emit EarningsWithdrawn(recipient, amount);
    }
    
    /**
     * @dev Withdraw training rewards
     * @param recipient Address to receive rewards
     * @param amount Amount to withdraw
     */
    function withdrawTrainingRewards(address recipient, uint256 amount) external onlyWhenNotPaused {
        require(recipient != address(0), InvalidRecipient());
        require(amount <= trainingRewards[trainingPool], InsufficientBalance());
        
        trainingRewards[trainingPool] -= amount;
        bool success = _transferSafe(recipient, amount);
        require(success, TransferFailed());
        
        emit EarningsWithdrawn(recipient, amount);
    }
    
    /**
     * @dev Withdraw treasury funds
     * @param recipient Address to receive funds
     * @param amount Amount to withdraw
     */
    function withdrawTreasury(address recipient, uint256 amount) external onlyWhenNotPaused {
        require(recipient != address(0), InvalidRecipient());
        require(amount <= treasuryBalance, InsufficientBalance());
        
        treasuryBalance -= amount;
        bool success = _transferSafe(recipient, amount);
        require(success, TransferFailed());
        
        emit EarningsWithdrawn(recipient, amount);
    }
    
    /**
     * @dev Set custom payment rate for a model
     * @param modelId Model identifier
     * @param rate Payment rate in USDT
     */
    function setModelPaymentRate(string calldata modelId, uint256 rate) external onlyOwner {
        require(rate >= MIN_PAYMENT, InvalidServiceFee());
        modelPaymentRates[modelId] = rate;
        emit PaymentRateUpdated(modelId, rate);
    }
    
    /**
     * @dev Update service fee rate
     * @param newRate New service fee rate (basis points)
     */
    function setServiceFee(uint256 newRate) external onlyOwner {
        require(newRate <= MAX_SERVICE_FEE, InvalidServiceFee());
        serviceFeeRate = newRate;
    }
    
    /**
     * @dev Update treasury wallet
     * @param newTreasury New treasury address
     */
    function setTreasuryWallet(address newTreasury) external onlyOwner {
        require(newTreasury != address(0), InvalidRecipient());
        treasuryWallet = newTreasury;
        emit TreasuryUpdated(newTreasury);
    }
    
    /**
     * @dev Update seed node pool
     * @param newPool New pool address
     */
    function setSeedNodePool(address newPool) external onlyOwner {
        require(newPool != address(0), InvalidRecipient());
        seedNodePool = newPool;
        emit SeedNodePoolUpdated(newPool);
    }
    
    /**
     * @dev Update training pool
     * @param newPool New pool address
     */
    function setTrainingPool(address newPool) external onlyOwner {
        require(newPool != address(0), InvalidRecipient());
        trainingPool = newPool;
        emit TrainingPoolUpdated(newPool);
    }
    
    /**
     * @dev Pause/unpause contract
     * @param _paused New paused state
     */
    function setPaused(bool _paused) external onlyOwner {
        paused = _paused;
        emit Paused(_paused);
    }
    
    /**
     * @dev Emergency withdrawal in case of contract issues
     */
    function emergencyWithdraw(address token, address recipient, uint256 amount) external onlyOwner {
        require(token != address(0), InvalidRecipient());
        require(recipient != address(0), InvalidRecipient());
        
        IERC20(token).safeTransfer(recipient, amount);
    }
    
    /**
     * @dev Get payment details for a query
     * @param queryId Query identifier
     */
    function getQueryPayment(bytes32 queryId) external view returns (
        address payer,
        uint256 amount,
        uint256 timestamp,
        bool completed,
        bool refunded,
        bytes32 modelId
    ) {
        QueryPayment storage payment = queryPayments[queryId];
        return (
            payment.payer,
            payment.amount,
            payment.timestamp,
            payment.completed,
            payment.refunded,
            payment.modelId
        );
    }
    
    /**
     * @dev Get model payment rate
     * @param modelId Model identifier
     */
    function getModelPaymentRate(string calldata modelId) external view returns (uint256) {
        return modelPaymentRates[modelId];
    }
    
    /**
     * @dev Get contract statistics
     */
    function getStats() external view returns (
        uint256 totalQueries,
        uint256 totalVolume,
        uint256 seedBalance,
        uint256 trainingBalance,
        uint256 treasury,
        uint256 contractBalance
    ) {
        return (
            totalQueriesProcessed,
            totalPayoutVolume,
            seedNodeEarnings[seedNodePool],
            trainingRewards[trainingPool],
            treasuryBalance,
            usdtToken.balanceOf(address(this))
        );
    }
    
    /**
     * @dev Internal transfer with safety checks
     */
    function _transferSafe(address recipient, uint256 amount) internal returns (bool) {
        uint256 balance = usdtToken.balanceOf(address(this));
        if (amount > balance) {
            return false;
        }
        
        usdtToken.safeTransfer(recipient, amount);
        return true;
    }
    
    /**
     * @dev Rescue tokens sent to contract by mistake
     */
    function rescueTokens(address token, address recipient, uint256 amount) external onlyOwner {
        require(token != address(0), InvalidRecipient());
        require(recipient != address(0), InvalidRecipient());
        
        IERC20(token).safeTransfer(recipient, amount);
    }
}
