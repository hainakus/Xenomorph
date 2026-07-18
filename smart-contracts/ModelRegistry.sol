// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title ModelRegistry
 * @dev Registry for AI models with versioning and metadata
 */
contract ModelRegistry is Ownable, ReentrancyGuard {
    struct Model {
        string name;
        string description;
        string version;
        bytes32 modelHash;
        address submitter;
        uint256 blockHeight;
        bool active;
        bool deprecated;
        uint256 totalQueries;
        uint256 totalEarnings;
    }
    
    mapping(bytes32 => Model) public models;
    mapping(address => bytes32[]) public submittedModels;
    mapping(string => bytes32[]) public modelsByName;
    
    bytes32[] public modelList;
    mapping(bytes32 => uint256) public modelIndex;
    
    uint256 public totalModels;
    uint256 public activeModels;
    
    // Model categories
    mapping(string => bytes32[]) public modelsByCategory;
    string[] public categories;
    
    // Access control
    mapping(address => bool) public authorizedSubmitters;
    mapping(address => bool) public authorizedVerifiers;
    
    // Verification
    mapping(bytes32 => bytes32) public modelVerificationHash;
    mapping(bytes32 => bool) public verifiedModels;
    
    // Pricing
    mapping(bytes32 => uint256) public customPaymentRates;
    
    // Events
    event ModelRegistered(bytes32 indexed modelId, string name, address indexed submitter);
    event ModelVerified(bytes32 indexed modelId, bytes32 verificationHash);
    event ModelDeprecated(bytes32 indexed modelId);
    event ModelActivated(bytes32 indexed modelId);
    event ModelDeactivated(bytes32 indexed modelId);
    event SubmitterAuthorized(address indexed submitter, bool authorized);
    event VerifierAuthorized(address indexed verifier, bool authorized);
    event CustomRateSet(bytes32 indexed modelId, uint256 rate);
    event CategoryAdded(string category);
    
    error ModelAlreadyExists();
    error ModelNotFound();
    error NotAuthorized();
    error InvalidModelHash();
    error ModelAlreadyVerified();
    error InvalidCategory();
    error InvalidAddress();
    
    modifier onlyAuthorizedSubmitter() {
        require(authorizedSubmitters[msg.sender], NotAuthorized());
        _;
    }
    
    modifier onlyAuthorizedVerifier() {
        require(authorizedVerifiers[msg.sender], NotAuthorized());
        _;
    }
    
    constructor() Ownable(msg.sender) {
        // Add default categories
        _addCategory("NLP");
        _addCategory("Vision");
        _addCategory("Audio");
        _addCategory("Multimodal");
    }
    
    /**
     * @dev Register a new model
     * @param modelId Unique identifier for the model
     * @param name Model name
     * @param description Model description
     * @param version Model version
     * @param modelHash Hash of model weights
     * @param category Model category
     */
    function registerModel(
        bytes32 modelId,
        string calldata name,
        string calldata description,
        string calldata version,
        bytes32 modelHash,
        string calldata category
    ) external onlyAuthorizedSubmitter nonReentrant {
        require(models[modelId].submitter == address(0), ModelAlreadyExists());
        require(_categoryExists(category), InvalidCategory());
        
        models[modelId] = Model({
            name: name,
            description: description,
            version: version,
            modelHash: modelHash,
            submitter: msg.sender,
            blockHeight: block.number,
            active: true,
            deprecated: false,
            totalQueries: 0,
            totalEarnings: 0
        });
        
        submittedModels[msg.sender].push(modelId);
        modelsByName[name].push(modelId);
        modelsByCategory[category].push(modelId);
        
        modelList.push(modelId);
        modelIndex[modelId] = modelList.length - 1;
        
        totalModels++;
        activeModels++;
        
        emit ModelRegistered(modelId, name, msg.sender);
    }
    
    /**
     * @dev Verify a model after training completion
     * @param modelId Model identifier
     * @param verificationHash Hash of the trained model
     */
    function verifyModel(bytes32 modelId, bytes32 verificationHash) external onlyAuthorizedVerifier nonReentrant {
        Model storage model = models[modelId];
        require(model.submitter != address(0), ModelNotFound());
        require(!verifiedModels[modelId], ModelAlreadyVerified());
        
        modelVerificationHash[modelId] = verificationHash;
        verifiedModels[modelId] = true;
        
        emit ModelVerified(modelId, verificationHash);
    }
    
    /**
     * @dev Deprecate a model
     * @param modelId Model identifier
     */
    function deprecateModel(bytes32 modelId) external onlyOwner {
        Model storage model = models[modelId];
        require(model.submitter != address(0), ModelNotFound());
        require(!model.deprecated, "Model already deprecated");
        
        model.deprecated = true;
        model.active = false;
        activeModels--;
        
        emit ModelDeprecated(modelId);
    }
    
    /**
     * @dev Activate a deprecated model
     * @param modelId Model identifier
     */
    function activateModel(bytes32 modelId) external onlyOwner {
        Model storage model = models[modelId];
        require(model.submitter != address(0), ModelNotFound());
        require(model.deprecated, "Model not deprecated");
        
        model.deprecated = false;
        model.active = true;
        activeModels++;
        
        emit ModelActivated(modelId);
    }
    
    /**
     * @dev Deactivate a model temporarily
     * @param modelId Model identifier
     */
    function deactivateModel(bytes32 modelId) external onlyOwner {
        Model storage model = models[modelId];
        require(model.submitter != address(0), ModelNotFound());
        require(model.active, "Model not active");
        
        model.active = false;
        activeModels--;
        
        emit ModelDeactivated(modelId);
    }
    
    /**
     * @dev Authorize a submitter
     * @param submitter Address to authorize
     * @param authorized Authorization status
     */
    function authorizeSubmitter(address submitter, bool authorized) external onlyOwner {
        require(submitter != address(0), InvalidAddress());
        authorizedSubmitters[submitter] = authorized;
        emit SubmitterAuthorized(submitter, authorized);
    }
    
    /**
     * @dev Authorize a verifier
     * @param verifier Address to authorize
     * @param authorized Authorization status
     */
    function authorizeVerifier(address verifier, bool authorized) external onlyOwner {
        require(verifier != address(0), InvalidAddress());
        authorizedVerifiers[verifier] = authorized;
        emit VerifierAuthorized(verifier, authorized);
    }
    
    /**
     * @dev Set custom payment rate for a model
     * @param modelId Model identifier
     * @param rate Payment rate in USDT
     */
    function setCustomRate(bytes32 modelId, uint256 rate) external onlyOwner {
        Model storage model = models[modelId];
        require(model.submitter != address(0), ModelNotFound());
        require(rate >= 1 * 10**6, "Rate too low"); // Minimum 1 USDT
        
        customPaymentRates[modelId] = rate;
        emit CustomRateSet(modelId, rate);
    }
    
    /**
     * @dev Increment query count for a model
     * @param modelId Model identifier
     */
    function incrementQueryCount(bytes32 modelId) external onlyAuthorizedSubmitter {
        Model storage model = models[modelId];
        require(model.submitter != address(0), ModelNotFound());
        
        model.totalQueries++;
    }
    
    /**
     * @dev Add earnings to a model
     * @param modelId Model identifier
     * @param amount Earnings amount
     */
    function addEarnings(bytes32 modelId, uint256 amount) external onlyAuthorizedSubmitter {
        Model storage model = models[modelId];
        require(model.submitter != address(0), ModelNotFound());
        
        model.totalEarnings += amount;
    }
    
    /**
     * @dev Add a new category
     * @param category Category name
     */
    function addCategory(string calldata category) external onlyOwner {
        _addCategory(category);
    }
    
    function _addCategory(string memory category) internal {
        require(!_categoryExists(category), InvalidCategory());
        
        categories.push(category);
        emit CategoryAdded(category);
    }
    
    function _categoryExists(string memory category) internal view returns (bool) {
        for (uint i = 0; i < categories.length; i++) {
            if (keccak256(abi.encode(categories[i])) == keccak256(abi.encode(category))) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * @dev Get model details
     * @param modelId Model identifier
     */
    function getModel(bytes32 modelId) external view returns (
        string memory name,
        string memory description,
        string memory version,
        bytes32 modelHash,
        address submitter,
        uint256 blockHeight,
        bool active,
        bool deprecated,
        uint256 totalQueries,
        uint256 totalEarnings,
        bool verified
    ) {
        Model storage model = models[modelId];
        return (
            model.name,
            model.description,
            model.version,
            model.modelHash,
            model.submitter,
            model.blockHeight,
            model.active,
            model.deprecated,
            model.totalQueries,
            model.totalEarnings,
            verifiedModels[modelId]
        );
    }
    
    /**
     * @dev Get models by name
     * @param name Model name
     */
    function getModelsByName(string calldata name) external view returns (bytes32[] memory) {
        return modelsByName[name];
    }
    
    /**
     * @dev Get models by category
     * @param category Category name
     */
    function getModelsByCategory(string calldata category) external view returns (bytes32[] memory) {
        return modelsByCategory[category];
    }
    
    /**
     * @dev Get all model IDs
     */
    function getAllModelIds() external view returns (bytes32[] memory) {
        return modelList;
    }
    
    /**
     * @dev Get submitted models for an address
     * @param submitter Submitter address
     */
    function getSubmittedModels(address submitter) external view returns (bytes32[] memory) {
        return submittedModels[submitter];
    }
    
    /**
     * @dev Get custom payment rate for a model
     * @param modelId Model identifier
     */
    function getCustomRate(bytes32 modelId) external view returns (uint256) {
        return customPaymentRates[modelId];
    }
    
    /**
     * @dev Check if submitter is authorized
     * @param submitter Address to check
     */
    function isAuthorizedSubmitter(address submitter) external view returns (bool) {
        return authorizedSubmitters[submitter];
    }
    
    /**
     * @dev Check if verifier is authorized
     * @param verifier Address to check
     */
    function isAuthorizedVerifier(address verifier) external view returns (bool) {
        return authorizedVerifiers[verifier];
    }
    
    /**
     * @dev Get all categories
     */
    function getCategories() external view returns (string[] memory) {
        return categories;
    }
    
    /**
     * @dev Get registry statistics
     */
    function getStats() external view returns (
        uint256 _totalModels,
        uint256 _activeModels,
        uint256 _totalCategories
    ) {
        return (totalModels, activeModels, categories.length);
    }
}
