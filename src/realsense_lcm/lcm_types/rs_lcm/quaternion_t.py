"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class quaternion_t(object):
    __slots__ = ["x", "y", "z", "w"]

    __typenames__ = ["double", "double", "double", "double"]

    __dimensions__ = [None, None, None, None]

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 0.0

    def encode(self):
        buf = BytesIO()
        buf.write(quaternion_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">dddd", self.x, self.y, self.z, self.w))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != quaternion_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return quaternion_t._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = quaternion_t()
        self.x, self.y, self.z, self.w = struct.unpack(">dddd", buf.read(32))
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if quaternion_t in parents: return 0
        tmphash = (0x9b1dee9dfc8c0515) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff) + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if quaternion_t._packed_fingerprint is None:
            quaternion_t._packed_fingerprint = struct.pack(">Q", quaternion_t._get_hash_recursive([]))
        return quaternion_t._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

