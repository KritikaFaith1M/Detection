// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AuditNotary {

    struct AuditRecord {
        string cid;
        uint256 timestamp;
        address sender;
    }

    AuditRecord[] public records;

    event HashStored(string cid, uint256 timestamp, address sender);

    function storeHash(string memory _cid) public {
        records.push(AuditRecord(_cid, block.timestamp, msg.sender));
        emit HashStored(_cid, block.timestamp, msg.sender);
    }

    function getAuditCount() public view returns (uint256) {
        return records.length;
    }

    function getRecord(uint index) public view returns (
        string memory,
        uint256,
        address
    ) {
        AuditRecord memory r = records[index];
        return (r.cid, r.timestamp, r.sender);
    }
}