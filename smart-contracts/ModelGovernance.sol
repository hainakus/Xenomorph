// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title ModelGovernance
 * @notice Permissionless governance for adding and deprecating AI models.
 *
 * Anyone with enough stake can propose a new model. Stake-weighted votes run
 * for 7 days, and approved proposals can be executed permissionlessly after a
 * 1-day delay. Models can later be deprecated by the owner (or automatically
 * by the consensus layer through an owner call).
 */
contract ModelGovernance is ReentrancyGuard, Ownable {

    struct ModelProposal {
        string modelId;
        string hfRepo;
        string hfRevision;
        bytes32 genesisCheckpoint;
        uint256 vramRequired;
        uint256 rewardPerBlock;
        uint256 minStakeToTrain;
        uint256 voteStart;
        uint256 voteEnd;
        uint256 yesVotes;
        uint256 noVotes;
        bool executed;
        mapping(address => uint256) votes;
        mapping(address => bool) hasVoted;
    }

    struct ActiveModel {
        string modelId;
        string hfRepo;
        string hfRevision;
        bytes32 genesisCheckpoint;
        uint256 vramRequired;
        uint256 rewardPerBlock;
        uint256 minStakeToTrain;
        uint256 activationBlock;
        uint256 deprecationBlock;
        bool active;
    }

    uint256 public constant VOTING_PERIOD = 7 days;
    uint256 public constant EXECUTION_DELAY = 1 days;
    uint256 public constant PROPOSAL_THRESHOLD = 10_000_000_000_000; // 10k Xenom
    uint256 public constant QUORUM = 1_000_000_000_000_000;          // 1M Xenom
    uint256 public constant APPROVAL_THRESHOLD = 66;                   // 66%
    uint256 public constant MAX_ACTIVE_MODELS = 10;

    mapping(uint256 => ModelProposal) public proposals;
    mapping(string => ActiveModel) public activeModels;
    string[] public activeModelList;

    uint256 public proposalCount;

    // Simplified staking placeholder. Integrate with the real staking contract
    // when available; for governance tests this balance can be set directly.
    mapping(address => uint256) public stakeBalance;

    event ModelProposed(
        uint256 indexed proposalId,
        string modelId,
        address proposer,
        uint256 voteEnd
    );
    event VoteCast(
        uint256 indexed proposalId,
        address voter,
        bool support,
        uint256 weight
    );
    event ProposalExecuted(uint256 indexed proposalId);
    event ModelActivated(
        string modelId,
        uint256 activationBlock,
        uint256 rewardPerBlock
    );
    event ModelDeprecated(string modelId, uint256 blockNumber);

    error AlreadyVoted(uint256 proposalId, address voter);
    error AlreadyExecuted(uint256 proposalId);
    error InvalidModelId();
    error InvalidVRAM();
    error InvalidReward();
    error MaxModelsReached();
    error ModelExists(string modelId);
    error ModelNotActive(string modelId);
    error QuorumNotReached(uint256 totalVotes);
    error NotApproved(uint256 yesPercentage);
    error ExecutionDelayNotMet(uint256 availableAt);
    error VotingOngoing(uint256 proposalId);
    error VotingEnded(uint256 proposalId);
    error InsufficientStake(uint256 required, uint256 actual);

    constructor(address initialOwner) Ownable(initialOwner) {}

    /**
     * @notice Create a proposal to add a new model.
     */
    function proposeModel(
        string calldata modelId,
        string calldata hfRepo,
        string calldata hfRevision,
        bytes32 genesisCheckpoint,
        uint256 vramRequired,
        uint256 rewardPerBlock,
        uint256 minStakeToTrain
    ) external payable returns (uint256 proposalId) {
        if (bytes(modelId).length == 0) revert InvalidModelId();
        if (vramRequired == 0) revert InvalidVRAM();
        if (rewardPerBlock == 0) revert InvalidReward();
        if (getActiveModelCount() >= MAX_ACTIVE_MODELS) revert MaxModelsReached();
        if (activeModels[modelId].activationBlock != 0) revert ModelExists(modelId);
        if (stakeBalance[msg.sender] < PROPOSAL_THRESHOLD) {
            revert InsufficientStake(PROPOSAL_THRESHOLD, stakeBalance[msg.sender]);
        }

        proposalId = proposalCount++;
        ModelProposal storage p = proposals[proposalId];

        p.modelId = modelId;
        p.hfRepo = hfRepo;
        p.hfRevision = hfRevision;
        p.genesisCheckpoint = genesisCheckpoint;
        p.vramRequired = vramRequired;
        p.rewardPerBlock = rewardPerBlock;
        p.minStakeToTrain = minStakeToTrain;
        p.voteStart = block.timestamp;
        p.voteEnd = block.timestamp + VOTING_PERIOD;

        emit ModelProposed(proposalId, modelId, msg.sender, p.voteEnd);
    }

    /**
     * @notice Vote on a proposal with stake-weighted voting power.
     */
    function vote(uint256 proposalId, bool support) external {
        ModelProposal storage p = proposals[proposalId];

        if (block.timestamp >= p.voteEnd) revert VotingEnded(proposalId);
        if (p.executed) revert AlreadyExecuted(proposalId);
        if (p.hasVoted[msg.sender]) revert AlreadyVoted(proposalId, msg.sender);

        uint256 voterStake = stakeBalance[msg.sender];
        if (voterStake == 0) revert InsufficientStake(1, 0);

        p.votes[msg.sender] = voterStake;
        p.hasVoted[msg.sender] = true;

        if (support) {
            p.yesVotes += voterStake;
        } else {
            p.noVotes += voterStake;
        }

        emit VoteCast(proposalId, msg.sender, support, voterStake);
    }

    /**
     * @notice Execute an approved proposal and activate the model.
     */
    function executeProposal(uint256 proposalId) external nonReentrant {
        ModelProposal storage p = proposals[proposalId];

        if (block.timestamp <= p.voteEnd) revert VotingOngoing(proposalId);
        if (block.timestamp < p.voteEnd + EXECUTION_DELAY) {
            revert ExecutionDelayNotMet(p.voteEnd + EXECUTION_DELAY);
        }
        if (p.executed) revert AlreadyExecuted(proposalId);

        uint256 totalVotes = p.yesVotes + p.noVotes;
        if (totalVotes < QUORUM) revert QuorumNotReached(totalVotes);

        uint256 yesPercentage = (p.yesVotes * 100) / totalVotes;
        if (yesPercentage < APPROVAL_THRESHOLD) revert NotApproved(yesPercentage);

        _activateModel(p);
        p.executed = true;

        emit ProposalExecuted(proposalId);
    }

    /**
     * @notice Deprecate an active model. Only the owner can deprecate directly;
     * the consensus layer can also trigger this through an owner relay.
     */
    function deprecateModel(string calldata modelId) external onlyOwner {
        ActiveModel storage m = activeModels[modelId];
        if (!m.active) revert ModelNotActive(modelId);

        m.active = false;
        m.deprecationBlock = block.number;

        _removeFromActiveList(modelId);

        emit ModelDeprecated(modelId, block.number);
    }

    /**
     * @notice Return true if a model is active and valid for mining at `block`.
     */
    function isModelActive(string calldata modelId, uint256 blockNumber) external view returns (bool) {
        ActiveModel storage m = activeModels[modelId];
        return m.active && blockNumber >= m.activationBlock;
    }

    /**
     * @notice List all currently active models. Used by seed nodes.
     */
    function getActiveModels() external view returns (ActiveModel[] memory) {
        uint256 count = getActiveModelCount();
        ActiveModel[] memory result = new ActiveModel[](count);

        uint256 idx = 0;
        for (uint256 i = 0; i < activeModelList.length; i++) {
            string memory id = activeModelList[i];
            if (activeModels[id].active) {
                result[idx++] = activeModels[id];
            }
        }
        return result;
    }

    /**
     * @notice Return metadata for a specific model.
     */
    function getModelData(string calldata modelId) external view returns (ActiveModel memory) {
        require(activeModels[modelId].activationBlock > 0, "Model not found");
        return activeModels[modelId];
    }

    /**
     * @notice Return the number of active models.
     */
    function getActiveModelCount() public view returns (uint256) {
        uint256 count = 0;
        for (uint256 i = 0; i < activeModelList.length; i++) {
            if (activeModels[activeModelList[i]].active) count++;
        }
        return count;
    }

    /**
     * @notice Simplified stake. In production this should pull Xenom tokens.
     */
    function stake(uint256 amount) external {
        stakeBalance[msg.sender] += amount;
    }

    /**
     * @notice Simplified unstake. In production this should return Xenom tokens.
     */
    function unstake(uint256 amount) external {
        require(stakeBalance[msg.sender] >= amount, "Insufficient stake");
        stakeBalance[msg.sender] -= amount;
    }

    // Internal helpers

    function _activateModel(ModelProposal storage p) internal {
        require(!activeModels[p.modelId].active, "Already active");

        activeModels[p.modelId] = ActiveModel({
            modelId: p.modelId,
            hfRepo: p.hfRepo,
            hfRevision: p.hfRevision,
            genesisCheckpoint: p.genesisCheckpoint,
            vramRequired: p.vramRequired,
            rewardPerBlock: p.rewardPerBlock,
            minStakeToTrain: p.minStakeToTrain,
            activationBlock: block.number + 1,
            deprecationBlock: 0,
            active: true
        });

        activeModelList.push(p.modelId);

        emit ModelActivated(p.modelId, block.number + 1, p.rewardPerBlock);
    }

    function _removeFromActiveList(string memory modelId) internal {
        for (uint256 i = 0; i < activeModelList.length; i++) {
            if (keccak256(bytes(activeModelList[i])) == keccak256(bytes(modelId))) {
                activeModelList[i] = activeModelList[activeModelList.length - 1];
                activeModelList.pop();
                break;
            }
        }
    }
}
