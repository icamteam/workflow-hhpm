from aiida import orm


class HhpmJsonableStructureData(orm.StructureData):
    def as_dict(self):
        return {
            "cell": [[item for item in row] for row in self.cell],
            "kinds": [kind.get_raw() for kind in self.kinds],
            "pbc": list(self.pbc),
            "sites": [site.get_raw() for site in self.sites],
        }

    @classmethod
    def from_dict(cls, dict_):
        structure = cls(cell=dict_["cell"], pbc=dict_["pbc"])
        for k in dict_["kinds"]:
            structure.append_kind(orm.Kind(raw=k))
        for s in dict_["sites"]:
            structure.append_site(orm.Site(kind_name=s["kind_name"], position=s["position"]))

        return structure
