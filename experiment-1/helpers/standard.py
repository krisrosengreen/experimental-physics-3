def plt_standard_settings(plt):
    plt.rc("font", family=["Arial"]) # skifter skrifttype
    plt.rc("axes", labelsize=18)   # skriftstørrelse af `xlabel` og `ylabel`
    plt.rc("xtick", labelsize=12, top=True, direction="in")  # skriftstørrelse af ticks og viser ticks øverst
    plt.rc("ytick", labelsize=12, right=True, direction="in")
    plt.rc("axes", titlesize=22)
    plt.rc("legend", fontsize=14)